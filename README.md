# comment-lint

`comment-lint` is a Rust CLI that detects likely superfluous code comments across Go, Python, TypeScript, JavaScript, and Rust.

A comment is considered potentially superfluous when it mostly repeats what adjacent code already says (for example, `// increment counter` immediately above `counter++`) and does not add durable context like intent, tradeoff, invariants, caveats, or external references.

For end-to-end model training, dataset generation, ONNX export, and evaluation details, see [training/README.md](training/README.md).

## Table of contents

- [What the tool does today](#what-the-tool-does-today)
- [Supported languages and detection](#supported-languages-and-detection)
- [Install and build](#install-and-build)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [Scoring backends](#scoring-backends)
- [Output formats](#output-formats)
- [Exit codes and CI usage](#exit-codes-and-ci-usage)
- [Configuration](#configuration)
- [Training and model lifecycle](#training-and-model-lifecycle)
- [Architecture overview](#architecture-overview)
- [Developer verification](#developer-verification)
- [Limitations and non-goals](#limitations-and-non-goals)

## What the tool does today

At runtime, `comment-lint`:

1. Walks files/directories you pass on the CLI.
2. Detects language by file extension.
3. Parses source using tree-sitter.
4. Extracts comments and surrounding context.
5. Builds a `FeatureVector` per comment.
6. Scores comments with a selected scorer (`heuristic`, `ml`, or `ensemble`).
7. Applies threshold/confidence filters.
8. Emits findings in the requested format.

The tool is intentionally detection-only right now. It does not rewrite source files.

## Supported languages and detection

Language detection is extension-based:

| Language | Extensions |
| --- | --- |
| Go | `.go` |
| Python | `.py` |
| TypeScript | `.ts`, `.tsx` |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` |
| Rust | `.rs` |

Notes:

- Unsupported extensions are skipped.
- Non-UTF-8 files are skipped.
- Directory traversal is recursive.

## Install and build

### Prerequisites

- Rust toolchain (stable)
- Optional for ML scoring: ONNX-capable build via the workspace `ml` feature
- Optional for model training: Python 3.12 + `uv` (see [training/README.md](training/README.md))

### Build heuristic-only binary

```bash
cargo build -p comment-lint
```

### Build binary with ML/ensemble support

```bash
cargo build -p comment-lint --features ml
```

If you run `--scorer ml` or `--scorer ensemble` without compiling with `--features ml`, the CLI exits with an error.

## Quick start

### 1) Scan a path with default settings

```bash
./target/debug/comment-lint crates/comment-lint-core/tests/fixtures
```

### 2) Lower threshold and include doc comments

```bash
./target/debug/comment-lint \
  --threshold 0.45 \
  --min-confidence 0.0 \
  --include-doc-comments \
  crates/comment-lint-core/tests/fixtures
```

### 3) Scan multiple paths

```bash
./target/debug/comment-lint crates/comment-lint-core crates/comment-lint
```

## CLI reference

Positional arguments:

- `paths...` (required): Files or directories to scan

Options:

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `-f, --format <FORMAT>` | `text \| json \| github` | `text` | Output formatter |
| `-t, --threshold <FLOAT>` | float | config default (`0.6`) | Minimum score to report |
| `--min-confidence <FLOAT>` | float | config default (`0.0`) | Minimum confidence to report |
| `--include-doc-comments` | bool | `false` | Include doc comments in analysis |
| `--config <PATH>` | path | none | Explicit TOML config overlay |
| `--export-features` | bool | `false` | Emit feature JSONL for ML training |
| `--scorer <NAME>` | `heuristic \| ml \| ensemble` | `heuristic` | Scoring backend |
| `--model-path <PATH>` | path | none | ONNX model path (required for `ml`/`ensemble` unless provided in config) |

Help:

```bash
./target/debug/comment-lint --help
```

## Scoring backends

### `heuristic` (default)

- Uses weighted feature scoring from `[weights]` and `[negative]` config sections.
- No ML model required.

### `ml`

- Requires binary compiled with `--features ml`.
- Loads ONNX model through `comment-lint-ml`.
- Model path resolution:
  1. `--model-path` (highest priority)
  2. `[ml].model_path` in merged config
  3. otherwise error

### `ensemble`

- Requires binary compiled with `--features ml`.
- Combines heuristic and ML scores with a fixed ML weight (`0.6`) in current implementation.
- Uses same model path resolution as `ml`.

## Output formats

### `text`

Human-readable line output with Unicode star ratings and color (when connected to a terminal).

Score-to-star mapping:

| Score range | Stars |
| --- | --- |
| 0.6 – 0.7 | ★☆☆☆☆ |
| 0.7 – 0.8 | ★★☆☆☆ |
| 0.8 – 0.85 | ★★★☆☆ |
| 0.85 – 0.9 | ★★★★☆ |
| 0.9+ | ★★★★★ |

Example:

```text
src/main.rs:42:4 — 0.85 ★★★★☆ — "increment counter" — high_lexical_overlap (0.80), identifier_match

Summary: 12 comments scanned, 1 superfluous, 1 files
```

### `json`

JSONL output.

- One JSON object per finding.
- Final summary line includes `type: "summary"`.

Example finding:

```json
{"file":"src/main.rs","line":42,"column":4,"comment_text":"increment counter","score":0.85,"confidence":0.9,"language":"rust","reasons":["high_lexical_overlap (0.80)","identifier_match"]}
```

Example summary line:

```json
{"type":"summary","total_comments":1,"superfluous_count":1,"file_count":1}
```

### `github`

GitHub Actions workflow command annotations (`::warning ...`).

Example:

```text
::warning file=src/main.rs,line=42,col=4::Superfluous comment (score: 0.85, confidence: 0.90): high_lexical_overlap (0.80), identifier_match
```

Summary output is intentionally omitted in this mode.

### `--export-features`

When `--export-features` is set:

- Formatter switches to feature JSONL export.
- `threshold` is forced to `0.0`.
- `min_confidence` is forced to `0.0`.
- `include_doc_comments` is forced to `true`.
- No summary line is emitted.

Example record:

```json
{"file":"src/main.rs","line":42,"column":4,"language":"rust","comment_text":"increment counter","comment_kind":"line","heuristic_score":0.85,"heuristic_confidence":0.9,"features":{"token_overlap_jaccard":0.8,"identifier_substring_ratio":0.9,"comment_token_count":2,"is_doc_comment":false,"is_before_declaration":true,"is_inline":false,"adjacent_node_kind":"function_item","nesting_depth":0,"has_why_indicator":false,"has_external_ref":false,"imperative_verb_noun":true,"is_section_label":false,"contains_literal_values":false,"references_other_files":false,"references_specific_functions":false,"mirrors_data_structure":false,"comment_code_age_ratio":null}}
```

This is the main handoff format into the Python training pipeline documented in [training/README.md](training/README.md).

## Exit codes and CI usage

Exit behavior:

- `0`: no findings above filters
- `1`: findings emitted
- `2`: CLI/config/scorer/output error

CI pattern example (fail on findings):

```bash
./target/debug/comment-lint --format github src/
```

Because findings return exit code `1`, this naturally fails the step unless you explicitly tolerate it.

## Configuration

Configuration is TOML-based and layered.

Resolution precedence (low to high):

1. Compiled defaults (`Config::default()`)
2. `config/default.toml`
3. `comment-lint.toml` in current working directory
4. `--config <path>` explicit file
5. CLI overrides for `threshold`, `min_confidence`, `include_doc_comments`

Then, if `--export-features` is present, `threshold/min_confidence/include_doc_comments` are overridden for full capture.

### Config keys

```toml
[general]
threshold = 0.6
min_confidence = 0.0
include_doc_comments = false

[weights]
token_overlap_jaccard = 0.25
identifier_substring_ratio = 0.20
imperative_verb_noun = 0.15
is_section_label = 0.10
contains_literal_values = 0.05
references_other_files = 0.05
mirrors_data_structure = 0.05

[negative]
has_why_indicator = -0.30
has_external_ref = -0.25
is_doc_comment_on_public = -0.20

[ignore]
paths = [
  "vendor/**",
  "node_modules/**",
  "*.generated.*",
  "target/**",
]
comment_patterns = [
  "^//go:generate",
  "^//nolint",
  "^# type:",
  "^# noqa",
  "^# pylint:",
  "^// eslint-",
  "^// @ts-",
  "^// Copyright",
  "^# Copyright",
]

[cache]
enabled = true
directory = ".comment-lint-cache"

[ml]
# Optional; can be overridden by --model-path
model_path = "training/data/model.onnx"
```

### Practical config examples

Custom threshold and confidence:

```toml
[general]
threshold = 0.7
min_confidence = 0.3
```

Use ML model by default:

```toml
[ml]
model_path = "training/data/model.onnx"
```

Run with config:

```bash
./target/debug/comment-lint --config comment-lint.toml src/
```

## Training and model lifecycle

Short version:

1. Export raw feature JSONL from real repositories with `--export-features`.
2. Build labeled datasets (heuristic labeling + optional manual review + synthetic/hq generation).
3. Train model and select best classifier.
4. Export ONNX with runtime contract validation.
5. Run `comment-lint --scorer ml` or `--scorer ensemble` with model path.

Detailed instructions, schemas, and command cookbook are in [training/README.md](training/README.md).

## Architecture overview

Workspace crates:

- `crates/comment-lint-core`: parsing, feature extraction, heuristic scoring, output formatters, pipeline orchestration
- `crates/comment-lint`: CLI entrypoint and config/formatter/scorer wiring
- `crates/comment-lint-ml`: ONNX tensor mapping, ML scorer, heuristic+ML ensemble scorer

Runtime data flow:

1. `Pipeline::discover_files` finds candidate files.
2. Language module extracts `CommentContext` using tree-sitter.
3. Core feature extractors build `FeatureVector`.
4. Selected scorer returns `ScoredComment`.
5. Filters (`threshold`, `min_confidence`) are applied.
6. Formatter emits output.

Key extension boundary:

- The `FeatureVector` fields are the contract between extraction and scoring.
- ML inference depends on fixed tensor ordering (documented in [training/README.md](training/README.md)).

## Developer verification

### CLI smoke checks

```bash
cargo run -p comment-lint -- --help
cargo run -p comment-lint -- crates/comment-lint-core/tests/fixtures
cargo run -p comment-lint -- --format json crates/comment-lint-core/tests/fixtures
```

ML build smoke checks:

```bash
cargo run -p comment-lint --features ml -- --help
cargo run -p comment-lint --features ml -- --scorer ml --model-path crates/comment-lint-ml/tests/fixtures/dummy_model.onnx crates/comment-lint-core/tests/fixtures
```

### Rust test suites

```bash
cargo test
cargo test -p comment-lint-core
cargo test -p comment-lint
cargo test -p comment-lint --features ml
cargo test -p comment-lint-ml
```

### Training package tests

See [training/README.md](training/README.md) for full test and command cookbook details.

```bash
cd training
uv sync --group dev
uv run pytest
```

## Limitations and non-goals

Current implementation limitations:

- Language detection is extension-based only (no shebang/content detection).
- Unsupported files are skipped, not force-parsed.
- The tool does not auto-fix or rewrite comments.
- The `cache` config section exists but is not currently used by the pipeline.
- `nesting_depth` is always `0` in current extraction (tree-sitter depth tracking not yet implemented).
- Runtime does not compute code-age metadata (`comment_code_age_ratio` is currently `None` in core extraction).

Non-goals in current implementation:

- LSP/editor diagnostics integration
- Diff-only or incremental mode
- Built-in model training (training is in `training/` Python package)
- Automatic comment rewriting
