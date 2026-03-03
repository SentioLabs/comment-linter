//! Pipeline orchestration: file discovery, parsing, feature extraction,
//! scoring, and streaming output.

use std::path::{Path, PathBuf};

use ignore::WalkBuilder;
use rayon::prelude::*;
use regex::Regex;
use walkdir::WalkDir;

use crate::config::Config;
use crate::features::cross_reference::extract_cross_reference_features;
use crate::features::lexical::extract_lexical_features;
use crate::features::semantic::extract_semantic_features;
use crate::features::structural::extract_structural_features;
use crate::features::{FeatureVector, ScoredComment};
use crate::languages::{detect_language, get_language};
use crate::scoring::Scorer;
use crate::types::CommentKind;

/// The result of a full pipeline run.
#[derive(Debug)]
pub struct PipelineResult {
    /// All comments that passed threshold and confidence filters.
    pub scored_comments: Vec<ScoredComment>,
    /// Total number of comments analyzed (before threshold/confidence filtering).
    pub total_comments_scanned: usize,
    /// Number of files successfully processed.
    pub files_processed: usize,
    /// Number of files skipped (unsupported extension, read error, etc.).
    pub files_skipped: usize,
}

/// Orchestrates file discovery, parsing, feature extraction, and scoring.
pub struct Pipeline {
    config: Config,
    scorer: Box<dyn Scorer + Send + Sync>,
    ignore_regexes: Vec<Regex>,
    comment_pattern_regexes: Vec<Regex>,
}

impl Pipeline {
    /// Create a new pipeline with the given configuration and scorer.
    pub fn new(config: Config, scorer: Box<dyn Scorer + Send + Sync>) -> Self {
        let ignore_regexes = config
            .ignore
            .paths
            .iter()
            .filter_map(|pattern| glob_to_regex(pattern).ok())
            .collect();

        let comment_pattern_regexes = config
            .ignore
            .comment_patterns
            .iter()
            .filter_map(|pattern| Regex::new(pattern).ok())
            .collect();

        Self {
            config,
            scorer,
            ignore_regexes,
            comment_pattern_regexes,
        }
    }

    /// Run the pipeline on the given paths (files or directories).
    ///
    /// Directories are walked recursively. Files with unsupported extensions
    /// or that match ignore patterns are skipped. Results are filtered by
    /// threshold and min_confidence from the config.
    pub fn run(&self, paths: &[PathBuf]) -> PipelineResult {
        let files = self.discover_files(paths);

        let threshold = self.config.general.threshold as f32;
        let min_confidence = self.config.general.min_confidence as f32;

        let results: Vec<Option<Vec<ScoredComment>>> = files
            .par_iter()
            .map(|path| self.process_file(path))
            .collect();

        let mut scored_comments = Vec::new();
        let mut total_comments_scanned = 0usize;
        let mut files_processed = 0usize;
        let mut files_skipped = 0usize;

        for result in results {
            match result {
                Some(comments) => {
                    files_processed += 1;
                    total_comments_scanned += comments.len();
                    scored_comments.extend(
                        comments
                            .into_iter()
                            .filter(|sc| sc.score >= threshold && sc.confidence >= min_confidence),
                    );
                }
                None => {
                    files_skipped += 1;
                }
            }
        }

        PipelineResult {
            scored_comments,
            total_comments_scanned,
            files_processed,
            files_skipped,
        }
    }

    /// Discover all files from the given paths, walking directories recursively.
    ///
    /// When `config.ignore.respect_gitignore` is true, uses [`ignore::WalkBuilder`]
    /// which automatically honours `.gitignore`, `.git/info/exclude`, and global
    /// gitignore rules. Otherwise falls back to plain `WalkDir`.
    fn discover_files(&self, paths: &[PathBuf]) -> Vec<PathBuf> {
        let mut files = Vec::new();
        let use_gitignore = self.config.ignore.respect_gitignore;

        for path in paths {
            if path.is_file() {
                files.push(path.clone());
            } else if path.is_dir() {
                if use_gitignore {
                    for entry in WalkBuilder::new(path).build().filter_map(|e| e.ok()) {
                        if entry.file_type().map_or(false, |ft| ft.is_file()) {
                            files.push(entry.into_path());
                        }
                    }
                } else {
                    for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
                        if entry.file_type().is_file() {
                            files.push(entry.into_path());
                        }
                    }
                }
            }
            // Non-existent paths are silently skipped
        }

        files
    }

    /// Process a single file: parse, extract comments, compute features, score.
    ///
    /// Returns `None` if the file cannot be processed (unsupported extension,
    /// read error, parse error, ignored, etc.).
    fn process_file(&self, path: &Path) -> Option<Vec<ScoredComment>> {
        // Check ignore patterns
        if self.is_ignored(path) {
            return None;
        }

        // Detect language
        let lang_id = detect_language(path)?;
        let lang = get_language(lang_id);

        // Read file as bytes, check UTF-8
        let bytes = std::fs::read(path).ok()?;
        let source = std::str::from_utf8(&bytes).ok()?;

        // Parse with tree-sitter
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&lang.tree_sitter_language()).ok()?;
        let tree = parser.parse(source, None)?;

        // Extract comments
        let comments = lang.extract_comments(&tree, source, path);

        // Process each comment: extract features, score
        let include_doc = self.config.general.include_doc_comments;

        let scored: Vec<ScoredComment> = comments
            .into_iter()
            .filter(|ctx| {
                // Skip doc comments if not included
                if !include_doc && ctx.comment_kind == CommentKind::Doc {
                    return false;
                }
                // Skip comments matching ignore patterns
                if self.is_comment_ignored(&ctx.comment_text) {
                    return false;
                }
                true
            })
            .map(|ctx| {
                let features = self.extract_features(&ctx);
                self.scorer.score(&ctx, &features)
            })
            .collect();

        Some(scored)
    }

    /// Extract the full feature vector from a comment context.
    fn extract_features(&self, ctx: &crate::extraction::comment::CommentContext) -> FeatureVector {
        let (jaccard, substr_ratio, token_count) = extract_lexical_features(ctx);
        let structural = extract_structural_features(ctx);
        let semantic =
            extract_semantic_features(&ctx.comment_text, &ctx.nearby_identifiers);
        let cross_ref =
            extract_cross_reference_features(&ctx.comment_text, &ctx.nearby_identifiers);

        FeatureVector {
            token_overlap_jaccard: jaccard,
            identifier_substring_ratio: substr_ratio,
            comment_token_count: token_count,
            is_doc_comment: structural.is_doc_comment,
            is_before_declaration: structural.is_before_declaration,
            is_inline: structural.is_inline,
            adjacent_node_kind: structural.adjacent_node_kind,
            nesting_depth: 0, // StructuralFeatures has no nesting_depth
            has_why_indicator: semantic.has_why_indicator,
            has_external_ref: semantic.has_external_ref,
            imperative_verb_noun: semantic.imperative_verb_noun,
            verb_noun_matches_identifier: semantic.verb_noun_matches_identifier,
            is_section_label: semantic.is_section_label,
            contains_literal_values: cross_ref.contains_literal_values,
            references_other_files: cross_ref.references_other_files,
            references_specific_functions: cross_ref.references_specific_functions,
            mirrors_data_structure: cross_ref.mirrors_data_structure,
            comment_code_age_ratio: None,
        }
    }

    /// Check if a file path matches any of the configured ignore patterns.
    fn is_ignored(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        self.ignore_regexes
            .iter()
            .any(|re| re.is_match(&path_str))
    }

    /// Check if a comment's text matches any of the configured comment ignore patterns.
    fn is_comment_ignored(&self, text: &str) -> bool {
        self.comment_pattern_regexes
            .iter()
            .any(|re| re.is_match(text))
    }
}

/// Convert a simple glob pattern to a regex.
///
/// Supports `*` (any characters except `/`) and `**` (any characters including `/`).
fn glob_to_regex(glob: &str) -> Result<Regex, regex::Error> {
    let mut regex_str = String::new();

    let chars: Vec<char> = glob.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            '*' => {
                if i + 1 < chars.len() && chars[i + 1] == '*' {
                    // ** matches everything including /
                    regex_str.push_str(".*");
                    i += 2;
                    // Skip trailing /
                    if i < chars.len() && chars[i] == '/' {
                        regex_str.push_str("/?");
                        i += 1;
                    }
                } else {
                    // * matches everything except /
                    regex_str.push_str("[^/]*");
                    i += 1;
                }
            }
            '?' => {
                regex_str.push_str("[^/]");
                i += 1;
            }
            '.' | '(' | ')' | '+' | '|' | '^' | '$' | '{' | '}' | '[' | ']' | '\\' => {
                regex_str.push('\\');
                regex_str.push(chars[i]);
                i += 1;
            }
            c => {
                regex_str.push(c);
                i += 1;
            }
        }
    }

    // The glob should match against path components, so wrap with anchoring
    // that allows matching at the end of a path or as a component
    Regex::new(&format!("(^|/){regex_str}($|/)"))
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::path::PathBuf;

    use tempfile::TempDir;

    use crate::config::Config;
    use crate::scoring::heuristic::HeuristicScorer;

    /// Helper: create a Pipeline with the given config.
    fn make_pipeline(config: Config) -> super::Pipeline {
        let scorer = HeuristicScorer::new(config.weights.clone(), config.negative.clone());
        super::Pipeline::new(config, Box::new(scorer))
    }

    /// Helper: write a file with given content into a temp dir.
    fn write_file(dir: &TempDir, name: &str, content: &str) -> PathBuf {
        let path = dir.path().join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create parent dirs");
        }
        let mut f = std::fs::File::create(&path).expect("create file");
        f.write_all(content.as_bytes()).expect("write file");
        path
    }

    // ---- Test: pipeline processes a single file and returns scored comments ----

    #[test]
    fn pipeline_processes_single_go_file() {
        let dir = tempfile::tempdir().expect("temp dir");
        let go_src = r#"package main

// increment the counter
func incrementCounter() {
    counter++
}
"#;
        let path = write_file(&dir, "main.go", go_src);

        let mut config = Config::default();
        config.general.threshold = 0.0;
        config.general.min_confidence = 0.0;
        config.general.include_doc_comments = true;
        config.ignore.paths = vec![];
        config.ignore.comment_patterns = vec![];

        let pipeline = make_pipeline(config);
        let result = pipeline.run(&[path]);

        assert!(
            !result.scored_comments.is_empty(),
            "Pipeline should produce at least one scored comment for a Go file with comments"
        );
    }

    // ---- Test: pipeline skips files with non-supported extensions ----

    #[test]
    fn pipeline_skips_unsupported_extensions() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write_file(&dir, "data.csv", "some,data,here");

        let mut config = Config::default();
        config.general.threshold = 0.0;
        config.ignore.paths = vec![];

        let pipeline = make_pipeline(config);
        let result = pipeline.run(&[path]);

        assert!(
            result.scored_comments.is_empty(),
            "Pipeline should return no scored comments for unsupported file extensions"
        );
    }

    // ---- Test: pipeline handles non-existent files gracefully ----

    #[test]
    fn pipeline_handles_nonexistent_files_gracefully() {
        let mut config = Config::default();
        config.ignore.paths = vec![];
        let pipeline = make_pipeline(config);
        let result = pipeline.run(&[PathBuf::from("/nonexistent/file.go")]);

        assert!(
            result.scored_comments.is_empty(),
            "Pipeline should return empty results for non-existent files"
        );
    }

    // ---- Test: pipeline filters results by threshold ----

    #[test]
    fn pipeline_filters_by_threshold() {
        let dir = tempfile::tempdir().expect("temp dir");
        let go_src = r#"package main

// increment counter
func incrementCounter() {
    counter++
}
"#;
        let path = write_file(&dir, "main.go", go_src);

        // Run with threshold=0.0 to see all results
        let mut config_low = Config::default();
        config_low.general.threshold = 0.0;
        config_low.general.min_confidence = 0.0;
        config_low.general.include_doc_comments = true;
        config_low.ignore.paths = vec![];
        config_low.ignore.comment_patterns = vec![];

        let pipeline_low = make_pipeline(config_low);
        let result_low = pipeline_low.run(&[path.clone()]);
        let count_low = result_low.scored_comments.len();

        // Run with threshold=1.0 -- should filter out everything
        let mut config_high = Config::default();
        config_high.general.threshold = 1.0;
        config_high.general.min_confidence = 0.0;
        config_high.general.include_doc_comments = true;
        config_high.ignore.paths = vec![];
        config_high.ignore.comment_patterns = vec![];

        let pipeline_high = make_pipeline(config_high);
        let result_high = pipeline_high.run(&[path]);
        let count_high = result_high.scored_comments.len();

        assert!(
            count_low > 0,
            "Low threshold should yield some scored comments"
        );
        assert!(
            count_high < count_low,
            "High threshold should yield fewer results than low threshold"
        );
    }

    // ---- Test: pipeline filters results by min_confidence ----

    #[test]
    fn pipeline_filters_by_min_confidence() {
        let dir = tempfile::tempdir().expect("temp dir");
        let go_src = r#"package main

// increment counter
func incrementCounter() {
    counter++
}
"#;
        let path = write_file(&dir, "main.go", go_src);

        // Run with min_confidence=0.0
        let mut config_low = Config::default();
        config_low.general.threshold = 0.0;
        config_low.general.min_confidence = 0.0;
        config_low.general.include_doc_comments = true;
        config_low.ignore.paths = vec![];
        config_low.ignore.comment_patterns = vec![];

        let pipeline_low = make_pipeline(config_low);
        let result_low = pipeline_low.run(&[path.clone()]);
        let count_low = result_low.scored_comments.len();

        // Run with min_confidence=1.0 -- should filter everything
        let mut config_high = Config::default();
        config_high.general.threshold = 0.0;
        config_high.general.min_confidence = 1.0;
        config_high.general.include_doc_comments = true;
        config_high.ignore.paths = vec![];
        config_high.ignore.comment_patterns = vec![];

        let pipeline_high = make_pipeline(config_high);
        let result_high = pipeline_high.run(&[path]);
        let count_high = result_high.scored_comments.len();

        assert!(
            count_low > 0,
            "Low min_confidence should yield some scored comments"
        );
        assert_eq!(
            count_high, 0,
            "min_confidence=1.0 should filter out all comments (confidence is never exactly 1.0)"
        );
    }

    // ---- Test: pipeline skips doc comments when include_doc_comments is false ----

    #[test]
    fn pipeline_skips_doc_comments_when_not_included() {
        let dir = tempfile::tempdir().expect("temp dir");
        let rust_src = r#"/// This is a doc comment
fn documented_function() {}

// This is a line comment
fn another_function() {}
"#;
        let path = write_file(&dir, "lib.rs", rust_src);

        // include_doc_comments = false (default)
        let mut config_no_doc = Config::default();
        config_no_doc.general.threshold = 0.0;
        config_no_doc.general.min_confidence = 0.0;
        config_no_doc.general.include_doc_comments = false;
        config_no_doc.ignore.paths = vec![];
        config_no_doc.ignore.comment_patterns = vec![];

        let pipeline_no_doc = make_pipeline(config_no_doc);
        let result_no_doc = pipeline_no_doc.run(&[path.clone()]);

        // include_doc_comments = true
        let mut config_with_doc = Config::default();
        config_with_doc.general.threshold = 0.0;
        config_with_doc.general.min_confidence = 0.0;
        config_with_doc.general.include_doc_comments = true;
        config_with_doc.ignore.paths = vec![];
        config_with_doc.ignore.comment_patterns = vec![];

        let pipeline_with_doc = make_pipeline(config_with_doc);
        let result_with_doc = pipeline_with_doc.run(&[path]);

        // No doc comments should be present when include_doc_comments is false
        for sc in &result_no_doc.scored_comments {
            assert_ne!(
                sc.context.comment_kind,
                crate::types::CommentKind::Doc,
                "Doc comments should not appear when include_doc_comments is false"
            );
        }

        // When doc comments are included, there should be at least one
        let has_doc = result_with_doc
            .scored_comments
            .iter()
            .any(|sc| sc.context.comment_kind == crate::types::CommentKind::Doc);
        assert!(
            has_doc || result_with_doc.scored_comments.len() > result_no_doc.scored_comments.len(),
            "Including doc comments should yield doc-kind results or more total results"
        );
    }

    // ---- Test: pipeline processes directory recursively ----

    #[test]
    fn pipeline_processes_directory_recursively() {
        let dir = tempfile::tempdir().expect("temp dir");

        let go_src_root = r#"package main

// increment counter
func incrementCounter() {}
"#;
        let go_src_nested = r#"package sub

// decrement counter
func decrementCounter() {}
"#;
        write_file(&dir, "root.go", go_src_root);
        write_file(&dir, "sub/nested.go", go_src_nested);

        let mut config = Config::default();
        config.general.threshold = 0.0;
        config.general.min_confidence = 0.0;
        config.general.include_doc_comments = true;
        config.ignore.paths = vec![];
        config.ignore.comment_patterns = vec![];

        let pipeline = make_pipeline(config);
        let result = pipeline.run(&[dir.path().to_path_buf()]);

        assert!(
            result.scored_comments.len() >= 2,
            "Pipeline should find comments in both root and nested files, found {}",
            result.scored_comments.len()
        );

        let has_root = result
            .scored_comments
            .iter()
            .any(|sc| sc.context.file_path.to_string_lossy().contains("root.go"));
        let has_nested = result
            .scored_comments
            .iter()
            .any(|sc| sc.context.file_path.to_string_lossy().contains("nested.go"));
        assert!(has_root, "Should find comments from root.go");
        assert!(has_nested, "Should find comments from nested.go");
    }

    // ---- Tests: .gitignore-aware file discovery ----

    /// Helper: initialise a bare git repo so that `ignore::WalkBuilder` recognises
    /// the directory as a git working tree and reads `.gitignore`.
    fn git_init(dir: &TempDir) {
        std::process::Command::new("git")
            .args(["init", "--quiet"])
            .current_dir(dir.path())
            .status()
            .expect("git init");
    }

    #[test]
    fn discover_files_respects_gitignore() {
        let dir = tempfile::tempdir().expect("temp dir");
        git_init(&dir);

        // .gitignore excludes the "generated/" directory
        write_file(&dir, ".gitignore", "generated/\n");

        let go_src = "package main\n\n// a comment\nfunc f() {}\n";
        write_file(&dir, "included.go", go_src);
        write_file(&dir, "generated/excluded.go", go_src);

        let mut config = Config::default();
        config.general.threshold = 0.0;
        config.general.min_confidence = 0.0;
        config.ignore.paths = vec![];
        config.ignore.comment_patterns = vec![];
        config.ignore.respect_gitignore = true;

        let pipeline = make_pipeline(config);
        let files = pipeline.discover_files(&[dir.path().to_path_buf()]);

        let has_included = files.iter().any(|p| p.to_string_lossy().contains("included.go"));
        let has_excluded = files.iter().any(|p| p.to_string_lossy().contains("excluded.go"));

        assert!(has_included, "included.go should be discovered");
        assert!(!has_excluded, "generated/excluded.go should be skipped by .gitignore");
    }

    #[test]
    fn discover_files_ignores_gitignore_when_disabled() {
        let dir = tempfile::tempdir().expect("temp dir");
        git_init(&dir);

        write_file(&dir, ".gitignore", "generated/\n");

        let go_src = "package main\n\n// a comment\nfunc f() {}\n";
        write_file(&dir, "included.go", go_src);
        write_file(&dir, "generated/excluded.go", go_src);

        let mut config = Config::default();
        config.general.threshold = 0.0;
        config.general.min_confidence = 0.0;
        config.ignore.paths = vec![];
        config.ignore.comment_patterns = vec![];
        config.ignore.respect_gitignore = false;

        let pipeline = make_pipeline(config);
        let files = pipeline.discover_files(&[dir.path().to_path_buf()]);

        let has_included = files.iter().any(|p| p.to_string_lossy().contains("included.go"));
        let has_excluded = files.iter().any(|p| p.to_string_lossy().contains("excluded.go"));

        assert!(has_included, "included.go should be discovered");
        assert!(has_excluded, "generated/excluded.go should also be discovered when gitignore is disabled");
    }
}
