//! Configuration system for comment-lint.
//!
//! Supports TOML-based configuration with layered resolution:
//! defaults -> config/default.toml -> comment-lint.toml -> explicit path.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub general: GeneralConfig,
    pub weights: WeightsConfig,
    pub negative: NegativeWeights,
    pub ignore: IgnoreConfig,
    pub cache: CacheConfig,
}

/// General settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GeneralConfig {
    pub threshold: f64,
    pub min_confidence: f64,
    pub include_doc_comments: bool,
}

/// Positive feature weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WeightsConfig {
    pub token_overlap_jaccard: f64,
    pub identifier_substring_ratio: f64,
    pub imperative_verb_noun: f64,
    pub is_section_label: f64,
    pub contains_literal_values: f64,
    pub references_other_files: f64,
    pub mirrors_data_structure: f64,
}

/// Negative feature weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NegativeWeights {
    pub has_why_indicator: f64,
    pub has_external_ref: f64,
    pub is_doc_comment_on_public: f64,
}

/// Ignore configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IgnoreConfig {
    pub paths: Vec<String>,
    pub comment_patterns: Vec<String>,
}

/// Cache configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheConfig {
    pub enabled: bool,
    pub directory: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            general: GeneralConfig::default(),
            weights: WeightsConfig::default(),
            negative: NegativeWeights::default(),
            ignore: IgnoreConfig::default(),
            cache: CacheConfig::default(),
        }
    }
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            threshold: 0.6,
            min_confidence: 0.0,
            include_doc_comments: false,
        }
    }
}

impl Default for WeightsConfig {
    fn default() -> Self {
        Self {
            token_overlap_jaccard: 0.25,
            identifier_substring_ratio: 0.20,
            imperative_verb_noun: 0.15,
            is_section_label: 0.10,
            contains_literal_values: 0.05,
            references_other_files: 0.05,
            mirrors_data_structure: 0.05,
        }
    }
}

impl Default for NegativeWeights {
    fn default() -> Self {
        Self {
            has_why_indicator: -0.30,
            has_external_ref: -0.25,
            is_doc_comment_on_public: -0.20,
        }
    }
}

impl Default for IgnoreConfig {
    fn default() -> Self {
        Self {
            paths: vec![
                "vendor/**".to_string(),
                "node_modules/**".to_string(),
                "*.generated.*".to_string(),
                "target/**".to_string(),
            ],
            comment_patterns: vec![
                "^//go:generate".to_string(),
                "^//nolint".to_string(),
                "^# type:".to_string(),
                "^# noqa".to_string(),
                "^# pylint:".to_string(),
                "^// eslint-".to_string(),
                "^// @ts-".to_string(),
                "^// Copyright".to_string(),
                "^# Copyright".to_string(),
            ],
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            directory: ".comment-lint-cache".to_string(),
        }
    }
}

impl Config {
    /// Parse a Config from a TOML string.
    pub fn from_toml_str(s: &str) -> Result<Config, toml::de::Error> {
        toml::from_str(s)
    }

    /// Load a Config from a TOML file.
    pub fn from_file(path: &Path) -> anyhow::Result<Config> {
        let content = std::fs::read_to_string(path)?;
        let cfg = Self::from_toml_str(&content)?;
        Ok(cfg)
    }

    /// Resolve configuration with layered merging.
    ///
    /// Resolution chain:
    /// 1. Start with compiled defaults
    /// 2. Overlay config/default.toml (relative to project_root)
    /// 3. Overlay comment-lint.toml (in project_root)
    /// 4. Overlay explicit_path (if provided)
    pub fn resolve(explicit_path: Option<&Path>, project_root: &Path) -> anyhow::Result<Config> {
        // Start with compiled-in defaults (the Default impl).
        let mut base = toml::Value::try_from(Config::default())?;

        // Layer 2: config/default.toml
        let default_toml = project_root.join("config").join("default.toml");
        if default_toml.exists() {
            let content = std::fs::read_to_string(&default_toml)?;
            let overlay: toml::Value = toml::from_str(&content)?;
            merge_toml(&mut base, &overlay);
        }

        // Layer 3: comment-lint.toml in project root
        let project_toml = project_root.join("comment-lint.toml");
        if project_toml.exists() {
            let content = std::fs::read_to_string(&project_toml)?;
            let overlay: toml::Value = toml::from_str(&content)?;
            merge_toml(&mut base, &overlay);
        }

        // Layer 4: explicit path
        if let Some(path) = explicit_path {
            let content = std::fs::read_to_string(path)?;
            let overlay: toml::Value = toml::from_str(&content)?;
            merge_toml(&mut base, &overlay);
        }

        // Deserialize the merged value back into Config.
        let cfg: Config = base.try_into()?;
        Ok(cfg)
    }
}

/// Recursively merge `overlay` into `base`. Tables are merged key-by-key;
/// all other value types in overlay replace the corresponding base value.
fn merge_toml(base: &mut toml::Value, overlay: &toml::Value) {
    match (base, overlay) {
        (toml::Value::Table(base_table), toml::Value::Table(overlay_table)) => {
            for (key, overlay_val) in overlay_table {
                if let Some(base_val) = base_table.get_mut(key) {
                    merge_toml(base_val, overlay_val);
                } else {
                    base_table.insert(key.clone(), overlay_val.clone());
                }
            }
        }
        (base, overlay) => {
            *base = overlay.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_default_threshold() {
        let cfg = Config::default();
        assert!(
            (cfg.general.threshold - 0.6).abs() < f64::EPSILON,
            "Expected default threshold 0.6, got {}",
            cfg.general.threshold
        );
    }

    #[test]
    fn test_default_weights() {
        let cfg = Config::default();
        assert!(
            (cfg.weights.token_overlap_jaccard - 0.25).abs() < f64::EPSILON,
            "Expected token_overlap_jaccard 0.25, got {}",
            cfg.weights.token_overlap_jaccard
        );
        assert!(
            (cfg.weights.identifier_substring_ratio - 0.20).abs() < f64::EPSILON,
            "Expected identifier_substring_ratio 0.20, got {}",
            cfg.weights.identifier_substring_ratio
        );
        assert!(
            (cfg.weights.imperative_verb_noun - 0.15).abs() < f64::EPSILON,
            "Expected imperative_verb_noun 0.15, got {}",
            cfg.weights.imperative_verb_noun
        );
    }

    #[test]
    fn test_default_negative_weights() {
        let cfg = Config::default();
        assert!(
            (cfg.negative.has_why_indicator - (-0.30)).abs() < f64::EPSILON,
            "Expected has_why_indicator -0.30, got {}",
            cfg.negative.has_why_indicator
        );
        assert!(
            (cfg.negative.has_external_ref - (-0.25)).abs() < f64::EPSILON,
            "Expected has_external_ref -0.25, got {}",
            cfg.negative.has_external_ref
        );
    }

    #[test]
    fn test_default_ignore_paths() {
        let cfg = Config::default();
        assert!(
            cfg.ignore.paths.contains(&"vendor/**".to_string()),
            "Expected vendor/** in default ignore paths"
        );
        assert!(
            cfg.ignore.paths.contains(&"node_modules/**".to_string()),
            "Expected node_modules/** in default ignore paths"
        );
        assert!(
            cfg.ignore.paths.contains(&"target/**".to_string()),
            "Expected target/** in default ignore paths"
        );
    }

    #[test]
    fn test_default_cache() {
        let cfg = Config::default();
        assert!(cfg.cache.enabled, "Expected cache enabled by default");
        assert_eq!(
            cfg.cache.directory, ".comment-lint-cache",
            "Expected default cache directory"
        );
    }

    #[test]
    fn test_from_toml_str_full() {
        let toml_str = r#"
[general]
threshold = 0.8
min_confidence = 0.1
include_doc_comments = true

[weights]
token_overlap_jaccard = 0.30

[negative]
has_why_indicator = -0.50

[ignore]
paths = ["custom/**"]
comment_patterns = ["^# custom"]

[cache]
enabled = false
directory = "/tmp/cache"
"#;
        let cfg = Config::from_toml_str(toml_str).expect("valid TOML");
        assert!((cfg.general.threshold - 0.8).abs() < f64::EPSILON);
        assert!((cfg.general.min_confidence - 0.1).abs() < f64::EPSILON);
        assert!(cfg.general.include_doc_comments);
        assert!((cfg.weights.token_overlap_jaccard - 0.30).abs() < f64::EPSILON);
        assert!((cfg.negative.has_why_indicator - (-0.50)).abs() < f64::EPSILON);
        assert_eq!(cfg.ignore.paths, vec!["custom/**"]);
        assert!(!cfg.cache.enabled);
        assert_eq!(cfg.cache.directory, "/tmp/cache");
    }

    #[test]
    fn test_empty_toml_gives_defaults() {
        let cfg = Config::from_toml_str("").expect("empty TOML is valid");
        assert!(
            (cfg.general.threshold - 0.6).abs() < f64::EPSILON,
            "Empty TOML should give default threshold 0.6, got {}",
            cfg.general.threshold
        );
        assert!(
            (cfg.weights.token_overlap_jaccard - 0.25).abs() < f64::EPSILON,
            "Empty TOML should give default token_overlap_jaccard"
        );
        assert!(
            cfg.cache.enabled,
            "Empty TOML should give cache enabled"
        );
    }

    #[test]
    fn test_partial_override() {
        let toml_str = r#"
[general]
threshold = 0.9
"#;
        let cfg = Config::from_toml_str(toml_str).expect("valid TOML");
        // Overridden value
        assert!(
            (cfg.general.threshold - 0.9).abs() < f64::EPSILON,
            "Threshold should be overridden to 0.9"
        );
        // Non-overridden values should keep defaults
        assert!(
            (cfg.weights.token_overlap_jaccard - 0.25).abs() < f64::EPSILON,
            "Non-overridden weights should keep defaults"
        );
        assert!(
            cfg.cache.enabled,
            "Non-overridden cache should keep default"
        );
    }

    #[test]
    fn test_from_file() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("test.toml");
        {
            let mut f = std::fs::File::create(&path).expect("create file");
            writeln!(f, "[general]\nthreshold = 0.75").expect("write");
        }
        let cfg = Config::from_file(&path).expect("load file");
        assert!(
            (cfg.general.threshold - 0.75).abs() < f64::EPSILON,
            "Expected threshold 0.75 from file"
        );
    }

    #[test]
    fn test_from_file_not_found() {
        let result = Config::from_file(Path::new("/nonexistent/path.toml"));
        assert!(result.is_err(), "Should error on missing file");
    }

    #[test]
    fn test_resolve_defaults_only() {
        // Use a temp dir with no config files so only compiled defaults apply.
        let dir = tempfile::tempdir().expect("temp dir");
        let cfg = Config::resolve(None, dir.path()).expect("resolve");
        assert!(
            (cfg.general.threshold - 0.6).abs() < f64::EPSILON,
            "Resolve with no files should give default threshold"
        );
    }

    #[test]
    fn test_resolve_project_config_overrides() {
        let dir = tempfile::tempdir().expect("temp dir");
        let cfg_path = dir.path().join("comment-lint.toml");
        {
            let mut f = std::fs::File::create(&cfg_path).expect("create file");
            writeln!(f, "[general]\nthreshold = 0.85").expect("write");
        }
        let cfg = Config::resolve(None, dir.path()).expect("resolve");
        assert!(
            (cfg.general.threshold - 0.85).abs() < f64::EPSILON,
            "Project comment-lint.toml should override default threshold"
        );
    }

    #[test]
    fn test_resolve_explicit_overrides_project() {
        let dir = tempfile::tempdir().expect("temp dir");
        // Project config sets 0.85
        let project_cfg = dir.path().join("comment-lint.toml");
        {
            let mut f = std::fs::File::create(&project_cfg).expect("create");
            writeln!(f, "[general]\nthreshold = 0.85").expect("write");
        }
        // Explicit config sets 0.95
        let explicit_cfg = dir.path().join("explicit.toml");
        {
            let mut f = std::fs::File::create(&explicit_cfg).expect("create");
            writeln!(f, "[general]\nthreshold = 0.95").expect("write");
        }
        let cfg = Config::resolve(Some(&explicit_cfg), dir.path()).expect("resolve");
        assert!(
            (cfg.general.threshold - 0.95).abs() < f64::EPSILON,
            "Explicit path should override project config"
        );
    }
}
