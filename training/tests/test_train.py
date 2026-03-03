"""Tests for train.py - model training pipeline."""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from clt.utils import FEATURE_NAMES

TRAINING_DIR = os.path.join(os.path.dirname(__file__), "..")


def _make_sample(label, **overrides):
    """Create a sample JSONL record with given label and optional feature overrides."""
    features = {
        "token_overlap_jaccard": 0.5,
        "identifier_substring_ratio": 0.3,
        "comment_token_count": 5,
        "is_doc_comment": False,
        "is_before_declaration": True,
        "is_inline": False,
        "nesting_depth": 0,
        "has_why_indicator": False,
        "has_external_ref": False,
        "imperative_verb_noun": False,
        "is_section_label": False,
        "contains_literal_values": False,
        "references_other_files": False,
        "references_specific_functions": False,
        "mirrors_data_structure": False,
        "comment_code_age_ratio": None,
    }
    features.update(overrides)
    return {"features": features, "label": label}


def _write_jsonl(path, records):
    """Write a list of records as JSONL."""
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _make_training_data(n_per_class=30):
    """Generate synthetic training data with clear separation between classes."""
    records = []
    for _ in range(n_per_class):
        # Superfluous comments: high overlap, identifier match, imperative verbs
        records.append(_make_sample(
            1,
            token_overlap_jaccard=0.85 + np.random.uniform(0, 0.15),
            identifier_substring_ratio=0.7 + np.random.uniform(0, 0.3),
            imperative_verb_noun=True,
            comment_token_count=3,
            is_before_declaration=True,
        ))
    for _ in range(n_per_class):
        # Valuable comments: low overlap, why indicators, external refs
        records.append(_make_sample(
            0,
            token_overlap_jaccard=0.05 + np.random.uniform(0, 0.15),
            identifier_substring_ratio=0.05 + np.random.uniform(0, 0.15),
            has_why_indicator=True,
            has_external_ref=True,
            comment_token_count=15,
            is_before_declaration=False,
        ))
    np.random.shuffle(records)
    return records


# --- Feature extraction tests ---


class TestFeatureExtraction:
    """Test that features are extracted in the correct order matching tensor.rs."""

    def test_extract_features_returns_16_columns(self):
        """Feature extraction should produce exactly 16 features."""
        from clt.train import extract_features

        records = [_make_sample(1), _make_sample(0)]
        X, y = extract_features(records)
        assert X.shape[1] == 16

    def test_extract_features_correct_order(self):
        """Features must be in the exact order defined by tensor.rs."""
        from clt.train import extract_features

        record = _make_sample(
            1,
            token_overlap_jaccard=0.91,
            identifier_substring_ratio=0.8,
            comment_token_count=5,
            is_doc_comment=False,
            is_before_declaration=True,
            is_inline=False,
            nesting_depth=2,
            has_why_indicator=False,
            has_external_ref=False,
            imperative_verb_noun=True,
            is_section_label=False,
            contains_literal_values=False,
            references_other_files=False,
            references_specific_functions=False,
            mirrors_data_structure=False,
            comment_code_age_ratio=None,
        )
        X, y = extract_features([record])
        row = X[0]

        assert row[0] == pytest.approx(0.91)   # token_overlap_jaccard
        assert row[1] == pytest.approx(0.8)    # identifier_substring_ratio
        assert row[2] == pytest.approx(5.0)    # comment_token_count
        assert row[3] == pytest.approx(0.0)    # is_doc_comment (False -> 0)
        assert row[4] == pytest.approx(1.0)    # is_before_declaration (True -> 1)
        assert row[5] == pytest.approx(0.0)    # is_inline (False -> 0)
        assert row[6] == pytest.approx(2.0)    # nesting_depth
        assert row[7] == pytest.approx(0.0)    # has_why_indicator
        assert row[8] == pytest.approx(0.0)    # has_external_ref
        assert row[9] == pytest.approx(1.0)    # imperative_verb_noun (True -> 1)
        assert row[10] == pytest.approx(0.0)   # is_section_label
        assert row[11] == pytest.approx(0.0)   # contains_literal_values
        assert row[12] == pytest.approx(0.0)   # references_other_files
        assert row[13] == pytest.approx(0.0)   # references_specific_functions
        assert row[14] == pytest.approx(0.0)   # mirrors_data_structure
        assert row[15] == pytest.approx(0.0)   # comment_code_age_ratio (null -> 0.0)

    def test_null_comment_code_age_ratio_maps_to_zero(self):
        """null comment_code_age_ratio should map to 0.0."""
        from clt.train import extract_features

        record = _make_sample(1, comment_code_age_ratio=None)
        X, _ = extract_features([record])
        assert X[0][15] == pytest.approx(0.0)

    def test_nonnull_comment_code_age_ratio_preserved(self):
        """Non-null comment_code_age_ratio should preserve value."""
        from clt.train import extract_features

        record = _make_sample(1, comment_code_age_ratio=0.75)
        X, _ = extract_features([record])
        assert X[0][15] == pytest.approx(0.75)

    def test_bool_true_maps_to_one(self):
        """Boolean True features should map to 1.0."""
        from clt.train import extract_features

        record = _make_sample(1, is_doc_comment=True, has_why_indicator=True)
        X, _ = extract_features([record])
        assert X[0][3] == pytest.approx(1.0)   # is_doc_comment
        assert X[0][7] == pytest.approx(1.0)   # has_why_indicator

    def test_bool_false_maps_to_zero(self):
        """Boolean False features should map to 0.0."""
        from clt.train import extract_features

        record = _make_sample(1, is_doc_comment=False, has_why_indicator=False)
        X, _ = extract_features([record])
        assert X[0][3] == pytest.approx(0.0)   # is_doc_comment
        assert X[0][7] == pytest.approx(0.0)   # has_why_indicator

    def test_labels_extracted_correctly(self):
        """Labels should be extracted as numpy array."""
        from clt.train import extract_features

        records = [_make_sample(1), _make_sample(0), _make_sample(1)]
        _, y = extract_features(records)
        np.testing.assert_array_equal(y, [1, 0, 1])

    def test_feature_names_constant_matches_tensor_rs(self):
        """FEATURE_NAMES constant should match the expected order from tensor.rs."""
        from clt.train import FEATURE_NAMES as module_names

        assert module_names == FEATURE_NAMES


# --- JSONL loading tests ---


class TestLoadJsonl:
    """Test JSONL file loading."""

    def test_load_jsonl_reads_records(self):
        """load_jsonl should read all records from a JSONL file."""
        from clt.train import load_jsonl

        records = [_make_sample(1), _make_sample(0)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
            path = f.name

        try:
            loaded = load_jsonl(path)
            assert len(loaded) == 2
            assert loaded[0]["label"] == 1
            assert loaded[1]["label"] == 0
        finally:
            os.unlink(path)

    def test_load_jsonl_skips_empty_lines(self):
        """load_jsonl should skip empty lines."""
        from clt.train import load_jsonl

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(_make_sample(1)) + "\n")
            f.write("\n")
            f.write(json.dumps(_make_sample(0)) + "\n")
            path = f.name

        try:
            loaded = load_jsonl(path)
            assert len(loaded) == 2
        finally:
            os.unlink(path)


# --- Training pipeline tests ---


class TestTrainModels:
    """Test the model training pipeline."""

    def test_train_models_returns_dict_of_models(self):
        """train_models should return a dict of trained model objects."""
        from clt.train import train_models, extract_features

        records = _make_training_data(n_per_class=30)
        X, y = extract_features(records)
        models = train_models(X, y)
        assert isinstance(models, dict)
        assert len(models) >= 1
        # Each model should have a predict_proba method
        for name, model in models.items():
            assert hasattr(model, "predict_proba"), f"{name} lacks predict_proba"

    def test_train_models_includes_expected_classifiers(self):
        """train_models should train LogisticRegression, RandomForest, and XGBClassifier."""
        from clt.train import train_models, extract_features

        records = _make_training_data(n_per_class=30)
        X, y = extract_features(records)
        models = train_models(X, y)
        expected_keys = {"LogisticRegression", "RandomForest", "XGBClassifier"}
        assert set(models.keys()) == expected_keys

    def test_select_best_model_picks_highest_f1(self):
        """select_best_model should select the model with the highest cross-val F1."""
        from clt.train import select_best_model, train_models, extract_features

        records = _make_training_data(n_per_class=30)
        X, y = extract_features(records)
        models = train_models(X, y)
        best_name, best_model, scores = select_best_model(models, X, y)
        assert best_name in models
        assert best_model is models[best_name]
        assert isinstance(scores, dict)
        # All models should have scores
        for name in models:
            assert name in scores


# --- CLI tests ---


class TestTrainCLI:
    """Test the train.py CLI behavior."""

    def test_help_flag_works(self):
        """--help should produce usage information and exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "clt.train", "--help"],
            capture_output=True,
            text=True,
            cwd=TRAINING_DIR,
        )
        assert result.returncode == 0
        assert "train" in result.stdout.lower() or "Train" in result.stdout

    def test_requires_train_argument(self):
        """Script should fail if --train is not provided."""
        result = subprocess.run(
            [sys.executable, "-m", "clt.train", "--output", "/tmp/model.joblib"],
            capture_output=True,
            text=True,
            cwd=TRAINING_DIR,
        )
        assert result.returncode != 0

    def test_end_to_end_training(self):
        """Full training run should produce a joblib model file."""
        from clt.train import extract_features, train_models

        records = _make_training_data(n_per_class=30)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as train_f:
            for record in records:
                train_f.write(json.dumps(record) + "\n")
            train_path = train_f.name

        output_path = tempfile.mktemp(suffix=".joblib")

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m", "clt.train",
                    "--train",
                    train_path,
                    "--output",
                    output_path,
                ],
                capture_output=True,
                text=True,
                cwd=TRAINING_DIR,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            assert os.path.exists(output_path), "Model file was not created"
        finally:
            os.unlink(train_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
