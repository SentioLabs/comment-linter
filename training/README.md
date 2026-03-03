# comment-lint Training Pipeline

This directory contains the Python training package used to build and evaluate the ONNX model consumed by `comment-lint` runtime ML scoring.

Runtime CLI usage and architecture live in [../README.md](../README.md). This document focuses on training data flow, model generation, and validation.

## Table of contents

- [Scope and runtime integration](#scope-and-runtime-integration)
- [Environment setup (uv)](#environment-setup-uv)
- [Pipeline at a glance](#pipeline-at-a-glance)
- [Command cookbook](#command-cookbook)
- [Data contracts](#data-contracts)
- [Feature index contract (Rust tensor parity)](#feature-index-contract-rust-tensor-parity)
- [High-quality dataset builder internals](#high-quality-dataset-builder-internals)
- [Training behavior](#training-behavior)
- [ONNX export contract](#onnx-export-contract)
- [Evaluation behavior](#evaluation-behavior)
- [End-to-end examples](#end-to-end-examples)
- [Reproducibility and artifact management](#reproducibility-and-artifact-management)
- [Troubleshooting](#troubleshooting)
- [Test suite and validation coverage](#test-suite-and-validation-coverage)

## Scope and runtime integration

The training package (`clt`) is responsible for:

1. Generating/exporting labeled datasets.
2. Splitting datasets for train/validation/test.
3. Training candidate classifiers.
4. Selecting the best model.
5. Exporting the selected model to ONNX with a strict runtime contract.
6. Evaluating ONNX output metrics and optional heuristic comparison.

`comment-lint` runtime integration points:

- Runtime scorer expects an ONNX model with:
  - Input tensor: `input` (`float32`, shape `[None, 16]`)
  - Output tensor: `probabilities` (`float32`, shape `[None, 2]`)
- The 16-feature ordering must match Rust tensor mapping in `crates/comment-lint-ml/src/tensor.rs`.

## Environment setup (uv)

Canonical workflow in this repo is `uv`.

### Prerequisites

- Python 3.12+
- `uv`
- Rust-built `comment-lint` binary if you use real-world feature export (`clt-generate-data`)

### Setup

```bash
cd training
uv sync --group dev
```

### Verify entrypoints

All `clt-*` commands support `--help`. Spot-check a few:

```bash
uv run clt-build-dataset --help
uv run clt-train --help
uv run clt-export-onnx --help
uv run clt-evaluate --help
```

If you need a local binary for feature export:

```bash
cd ..
cargo build -p comment-lint
```

## Pipeline at a glance

Text flow:

```text
comment-lint --export-features
  -> raw JSONL feature exports (unlabeled)
  -> auto labeling (heuristic score thresholds)
  -> optional manual review of uncertain labels
  -> optional synthetic + high-quality mixed corpus build
  -> stratified/grouped dataset splits
  -> model training + CV model selection
  -> ONNX export + contract validation
  -> ONNX evaluation + heuristic comparison
  -> runtime use in comment-lint --scorer ml|ensemble
```

Typical artifact flow:

```text
training/data/raw/*.jsonl
  -> training/data/labeled.jsonl
  -> training/data/splits/{train,val,test}.jsonl
  -> training/data/model.joblib
  -> training/data/model.onnx

or

training/data/hq/labeled_hq_<count>.jsonl
  -> training/data/hq/splits/{train,val,test}.jsonl
  -> training/data/hq/metadata.json
```

## Command cookbook

All commands below are run from `training/` with `uv run`.

### 1) `clt-generate-data`

Purpose:

- Runs `comment-lint --export-features <dir>`.
- Converts heuristic score to initial labels:
  - `score >= 0.7` -> `label=1` (`superfluous`)
  - `score <= 0.3` -> `label=0` (`valuable`)
  - otherwise -> `label=-1` (`uncertain`)

Required args:

- `--binary`: path to `comment-lint`
- `--dir`: repo/path to scan

Optional args:

- `--output`: output JSONL path (defaults to stdout)

Example:

```bash
uv run clt-generate-data \
  --binary ../target/debug/comment-lint \
  --dir ../crates \
  --output data/labeled.jsonl
```

### 2) `clt-generate-synthetic`

Purpose:

- Generates synthetic labeled examples across all supported languages.

Args:

- `--output` (required)
- `--count` (default `8000`)
- `--superfluous-ratio` (default `0.40`)
- `--seed` (default `42`)

Example:

```bash
uv run clt-generate-synthetic \
  --output data/synthetic.jsonl \
  --count 12000 \
  --superfluous-ratio 0.38 \
  --seed 42
```

### 3) `clt-build-dataset`

Purpose:

- Builds a high-quality mixed dataset using real-world strict/hard-case labels plus synthetic fill to meet language/label targets.
- Writes full dataset, grouped splits, and metadata report.

Args (key):

- `--raw-dir` (default `data/raw`)
- `--labeled-input` (default `data/labeled.jsonl`)
- `--output-dir` (default `data/hq`)
- `--count` (default `75000`)
- `--superfluous-ratio` (default `0.35`)
- `--real-fraction` (default `0.35`)
- `--train-ratio` (default `0.70`)
- `--val-ratio` (default `0.15`)
- `--seed` (default `42`)

Example:

```bash
uv run clt-build-dataset \
  --raw-dir data/raw \
  --labeled-input data/labeled.jsonl \
  --output-dir data/hq \
  --count 75000 \
  --superfluous-ratio 0.35 \
  --real-fraction 0.35 \
  --seed 42
```

Outputs:

- `data/hq/labeled_hq_<count>.jsonl`
- `data/hq/splits/train.jsonl`
- `data/hq/splits/val.jsonl`
- `data/hq/splits/test.jsonl`
- `data/hq/metadata.json`

### 4) `clt-label-review`

Purpose:

- Interactive review for uncertain labels (`label=-1`).
- Updates labels to `0`/`1` with `label_source=manual_review`.

Args:

- `--input` (required)
- `--output` (required)

Interactive choices:

- `s`: superfluous (`1`)
- `v`: valuable (`0`)
- `skip`: leave unchanged
- `q`: quit and save progress

Example:

```bash
uv run clt-label-review \
  --input data/labeled.jsonl \
  --output data/labeled.reviewed.jsonl
```

### 5) `clt-split-dataset`

Purpose:

- Merges one or more labeled JSONL files.
- Excludes `label=-1` records.
- Produces stratified train/val/test splits by label.

Args:

- `--input` (required, accepts multiple files)
- `--output-dir` (required)
- `--train-ratio` (default `0.70`)
- `--val-ratio` (default `0.15`)
- `--seed` (default `42`)

Example:

```bash
uv run clt-split-dataset \
  --input data/labeled.reviewed.jsonl data/synthetic.jsonl \
  --output-dir data/splits \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --seed 42
```

### 6) `clt-train`

Purpose:

- Loads one or more training files.
- Trains model candidates.
- Reports CV metrics.
- Selects best model by mean CV F1.
- Saves best model as joblib.

Args:

- `--train` (required, one or more files)
- `--val` (optional, one or more files)
- `--output` (required)

Example:

```bash
uv run clt-train \
  --train data/splits/train.jsonl \
  --val data/splits/val.jsonl \
  --output data/model.joblib
```

### 7) `clt-export-onnx`

Purpose:

- Converts trained joblib model to ONNX.
- Renames/normalizes outputs to runtime contract.
- Validates ONNX IO names/shapes/dtypes and sample inference behavior.

Args:

- `--model` (required joblib path)
- `--output` (required ONNX path)

Example:

```bash
uv run clt-export-onnx \
  --model data/model.joblib \
  --output data/model.onnx
```

### 8) `clt-evaluate`

Purpose:

- Runs ONNX model inference on test JSONL.
- Reports classification metrics and confusion matrix.
- Compares model vs heuristic when `heuristic_score` exists in records.

Args:

- `--model` (required ONNX path)
- `--test` (required test JSONL)

Example:

```bash
uv run clt-evaluate \
  --model data/model.onnx \
  --test data/splits/test.jsonl
```

Program behavior:

- Human-readable metrics are printed to stderr.
- Structured JSON metrics object is printed to stdout.

### 9) `clt-create-dummy-model`

Purpose:

- Creates a minimal ONNX model that satisfies the runtime MLScorer contract.
- Used for Rust integration tests (`crates/comment-lint-ml/tests/fixtures/dummy_model.onnx`).

Args:

- `--output` (default: `crates/comment-lint-ml/tests/fixtures/dummy_model.onnx`)

Example:

```bash
uv run clt-create-dummy-model --output crates/comment-lint-ml/tests/fixtures/dummy_model.onnx
```

## Data contracts

### Common JSONL shape (raw export)

Records exported by runtime `comment-lint --export-features` include:

```json
{
  "file": "/abs/or/relative/path",
  "line": 42,
  "column": 0,
  "language": "rust",
  "comment_text": "increment counter",
  "comment_kind": "line",
  "heuristic_score": 0.55,
  "heuristic_confidence": 1.0,
  "features": {
    "token_overlap_jaccard": 1.0,
    "identifier_substring_ratio": 1.0,
    "comment_token_count": 2,
    "is_doc_comment": false,
    "is_before_declaration": true,
    "is_inline": false,
    "adjacent_node_kind": "function_item",
    "nesting_depth": 0,
    "has_why_indicator": false,
    "has_external_ref": false,
    "imperative_verb_noun": false,
    "is_section_label": true,
    "contains_literal_values": false,
    "references_other_files": false,
    "references_specific_functions": false,
    "mirrors_data_structure": false,
    "comment_code_age_ratio": null
  }
}
```

### Label semantics and label sources

Label values:

- `0`: valuable comment
- `1`: superfluous comment
- `-1`: uncertain (needs review or exclusion)

Common `label_source` values encountered in this repo:

- `heuristic_auto`
- `uncertain`
- `manual_review`
- `fixture_ground_truth`
- `real_world_strict`
- `real_world_hard_rule`
- `synthetic_strong`
- `synthetic_hard`

### Split dataset shape

`clt-split-dataset` preserves existing fields and writes:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`

with `label=-1` excluded.

### High-quality dataset shape additions

`clt-build-dataset` output records include additional metadata fields:

- `dataset_tier` (for example: `gold`, `strict`, `hard_case`, `synthetic`)
- `repo_group` (group key used for anti-leakage split assignment)

It also writes `metadata.json` with aggregate distribution and split integrity data.

## Feature index contract (Rust tensor parity)

The training package uses `FEATURE_NAMES` order in `clt/utils.py`, which must match Rust tensor mapping in `crates/comment-lint-ml/src/tensor.rs`.

| Index | Feature name | Encoding |
| --- | --- | --- |
| 0 | `token_overlap_jaccard` | float |
| 1 | `identifier_substring_ratio` | float |
| 2 | `comment_token_count` | integer -> float |
| 3 | `is_doc_comment` | bool -> 0.0/1.0 |
| 4 | `is_before_declaration` | bool -> 0.0/1.0 |
| 5 | `is_inline` | bool -> 0.0/1.0 |
| 6 | `nesting_depth` | integer -> float |
| 7 | `has_why_indicator` | bool -> 0.0/1.0 |
| 8 | `has_external_ref` | bool -> 0.0/1.0 |
| 9 | `imperative_verb_noun` | bool -> 0.0/1.0 |
| 10 | `is_section_label` | bool -> 0.0/1.0 |
| 11 | `contains_literal_values` | bool -> 0.0/1.0 |
| 12 | `references_other_files` | bool -> 0.0/1.0 |
| 13 | `references_specific_functions` | bool -> 0.0/1.0 |
| 14 | `mirrors_data_structure` | bool -> 0.0/1.0 |
| 15 | `comment_code_age_ratio` | `None` -> `0.0`, else float |

Important:

- `adjacent_node_kind` is present in records but is not part of the 16-element tensor.
- Any reorder breaks runtime inference correctness.

## High-quality dataset builder internals

`clt-build-dataset` combines real and synthetic examples per `(language, label)` bucket.

### 1) Real-world candidate selection

From `--raw-dir` records:

- Strict superfluous candidate requires:
  - `heuristic_score >= 0.82`
  - strong superfluous rule evidence and margin
- Strict valuable candidate requires:
  - `heuristic_score <= 0.18`
  - strong valuable rule evidence and margin

Hard-case candidates are drawn from mid-confidence score range (`0.28` to `0.72`) when additional rule thresholds are met.

From `--labeled-input` records:

- Includes `fixture_ground_truth` and `manual_review` labeled records.

### 2) Quality filtering and normalization

The builder filters low-quality/noisy comments (empty separators, trivial tokens like bare `TODO`, generated headers, excessive length) and normalizes feature types/ranges.

### 3) Deduplication

Records are deduplicated by signature combining label, language, normalized text, and feature buckets to reduce leakage and repetitive duplicates.

### 4) Bucket balancing

Given target `count` and `superfluous_ratio`, each language gets a per-label target. The builder fills each bucket with:

1. Real-world samples up to `real_fraction` target (preferring gold/strict)
2. Synthetic samples for the remainder

### 5) Synthetic augmentation

Synthetic generation includes both strong and hard-case variants and mutates text/features to improve boundary coverage.

### 6) Group-leakage prevention

Records are split by `repo_group` so the same logical source group does not appear across train/val/test. Split integrity checks are written into `metadata.json`.

## Training behavior

`clt-train` currently trains three candidate classifiers:

- `LogisticRegression`
- `RandomForestClassifier`
- `XGBClassifier`

Training details:

- Class balancing is enabled (`class_weight="balanced"` where applicable).
- XGBoost uses `scale_pos_weight` derived from class counts.
- Cross-validation metrics (`accuracy`, `precision`, `recall`, `f1`) are reported for each model.
- Best model is selected by highest mean CV F1 (`select_best_model`).

Validation behavior:

- If `--val` is provided, a `classification_report` is printed on validation data for the selected model.

Output:

- Best model serialized to joblib at `--output`.

## ONNX export contract

`clt-export-onnx` enforces runtime compatibility.

Required ONNX interface:

- Input:
  - name: `input`
  - dtype: `float32`
  - shape: `[None, 16]`
- Output:
  - name: `probabilities`
  - dtype: `float32`
  - shape: `[None, 2]`

Output semantics:

- `probabilities[:, 0]` = `P(not_superfluous)`
- `probabilities[:, 1]` = `P(superfluous)`

Validation steps performed by exporter:

1. ONNX structural validation (`onnx.checker.check_model`).
2. Runtime session load.
3. Input name/dim checks.
4. Output name checks (`probabilities` exists).
5. Sample inference check for output shape/dtype.

Supported model input types for conversion:

- `LogisticRegression`
- `RandomForestClassifier`
- `XGBClassifier`

Unsupported model types fail fast with a clear error.

## Evaluation behavior

`clt-evaluate` computes:

- Accuracy
- Precision
- Recall
- F1
- Confusion matrix
- Full classification report

Heuristic comparison mode:

- If test records include `heuristic_score`, it computes heuristic predictions (`>= 0.5` -> class `1`) and reports heuristic vs model metrics on that subset.

Programmatic output:

- JSON object on stdout:
  - `metrics`
  - `heuristic_comparison`
  - `n_samples`

## Reproducibility and artifact management

### Seeds and deterministic behavior

Default seeds:

- `clt-generate-synthetic`: `--seed 42`
- `clt-split-dataset`: `--seed 42`
- `clt-build-dataset`: `--seed 42`

Model training randomness:

- Candidate models use fixed `random_state=42` where applicable.

### Recommended artifact layout

Use stable paths under `training/data/`:

```text
training/data/
  raw/*.jsonl
  labeled.jsonl
  synthetic.jsonl
  splits/
    train.jsonl
    val.jsonl
    test.jsonl
  hq/
    labeled_hq_<count>.jsonl
    metadata.json
    splits/
      train.jsonl
      val.jsonl
      test.jsonl
  model.joblib
  model.onnx
```

Recommended model handoff:

- Keep canonical runtime model at `training/data/model.onnx`.
- Reference this path with `--model-path` or `[ml].model_path` in runtime config (see [../README.md](../README.md)).

## End-to-end examples

### Standard pipeline (raw + review + split + train + export + eval)

```bash
# from repo root
cargo build -p comment-lint

# from training/
uv sync --group dev

uv run clt-generate-data \
  --binary ../target/debug/comment-lint \
  --dir ../crates \
  --output data/labeled.jsonl

uv run clt-label-review \
  --input data/labeled.jsonl \
  --output data/labeled.reviewed.jsonl

uv run clt-split-dataset \
  --input data/labeled.reviewed.jsonl \
  --output-dir data/splits \
  --seed 42

uv run clt-train \
  --train data/splits/train.jsonl \
  --val data/splits/val.jsonl \
  --output data/model.joblib

uv run clt-export-onnx \
  --model data/model.joblib \
  --output data/model.onnx

uv run clt-evaluate \
  --model data/model.onnx \
  --test data/splits/test.jsonl
```

### High-quality mixed corpus pipeline

```bash
uv run clt-build-dataset \
  --raw-dir data/raw \
  --labeled-input data/labeled.jsonl \
  --output-dir data/hq \
  --count 75000 \
  --superfluous-ratio 0.35 \
  --real-fraction 0.35 \
  --seed 42

uv run clt-train \
  --train data/hq/splits/train.jsonl \
  --val data/hq/splits/val.jsonl \
  --output data/model.joblib

uv run clt-export-onnx --model data/model.joblib --output data/model.onnx
uv run clt-evaluate --model data/model.onnx --test data/hq/splits/test.jsonl
```

## Troubleshooting

### `ml scorer is not available; rebuild with --features ml`

Cause:

- Runtime binary built without ML feature.

Fix:

```bash
cargo build -p comment-lint --features ml
```

### `ml scorer requires a model path ...`

Cause:

- No `--model-path` and no `[ml].model_path` configured.

Fix:

- Pass `--model-path training/data/model.onnx`, or configure `[ml].model_path`.

### `No labeled examples found (all uncertain?)`

Cause:

- `clt-split-dataset` excludes `label=-1` and none remained.

Fix:

- Review uncertain labels (`clt-label-review`) or add additional confidently labeled data.

### ONNX export fails with unsupported model type

Cause:

- Exporter currently supports only LR/RF/XGB classifier types.

Fix:

- Train using current `clt-train` model set or extend exporter code accordingly.

### ONNX runtime shape/name mismatch

Cause:

- Exported model does not match required IO contract.

Fix:

- Re-export with `clt-export-onnx` and ensure validation passes.

### `clt-generate-data` returns command failure from `comment-lint`

Cause:

- Wrong binary path or runtime parse failure.

Fix:

- Verify binary exists and runs:

```bash
../target/debug/comment-lint --help
```

### Data quality skew or weak model metrics

Cause:

- Class imbalance, noisy labels, or insufficient hard cases.

Fixes:

- Tune `--superfluous-ratio`, `--real-fraction`, and corpus size.
- Add curated/manual reviewed labels.
- Use high-quality dataset builder.

## Test suite and validation coverage

Run full suite:

```bash
cd training
uv sync --group dev
uv run pytest
```

Current tests cover:

- `test_utils.py`: feature constants, JSONL helpers, type coercion.
- `test_generate_data.py`: score-to-label mapping and CLI behavior.
- `test_split_dataset.py`: stratified split behavior and uncertain-label exclusion.
- `test_label_review.py`: interactive label review behavior.
- `test_train.py`: feature extraction order, model training and selection behavior.
- `test_export_onnx.py`: ONNX conversion and IO contract validation.
- `test_build_high_quality_dataset.py`: bucket targets, group-leakage split guarantees, quality filters.

For runtime CLI and Rust-side integration checks, use [../README.md](../README.md) developer verification commands.
