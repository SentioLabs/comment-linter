"""Tests for export_onnx.py - ONNX model export and contract validation."""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest

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


def _make_training_data(n_per_class=30):
    """Generate synthetic training data with clear class separation."""
    records = []
    for _ in range(n_per_class):
        records.append(_make_sample(
            1,
            token_overlap_jaccard=0.85 + np.random.uniform(0, 0.15),
            identifier_substring_ratio=0.7 + np.random.uniform(0, 0.3),
            imperative_verb_noun=True,
            comment_token_count=3,
        ))
    for _ in range(n_per_class):
        records.append(_make_sample(
            0,
            token_overlap_jaccard=0.05 + np.random.uniform(0, 0.15),
            identifier_substring_ratio=0.05 + np.random.uniform(0, 0.15),
            has_why_indicator=True,
            has_external_ref=True,
            comment_token_count=15,
        ))
    return records


def _train_and_save_model(output_path):
    """Train a model and save it to the given path, returning the model."""
    from train import extract_features, train_models, select_best_model
    import joblib

    records = _make_training_data(n_per_class=30)
    X, y = extract_features(records)
    models = train_models(X, y)
    best_name, best_model, _ = select_best_model(models, X, y)
    joblib.dump(best_model, output_path)
    return best_model


# --- ONNX contract tests ---


class TestOnnxExport:
    """Test ONNX export matches the required contract for MLScorer."""

    def test_export_produces_valid_onnx(self):
        """Exported model should be loadable by onnx.load and pass validation."""
        from export_onnx import export_to_onnx

        model_path = tempfile.mktemp(suffix=".joblib")
        onnx_path = tempfile.mktemp(suffix=".onnx")

        try:
            _train_and_save_model(model_path)
            export_to_onnx(model_path, onnx_path)

            # Must be loadable and valid
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    def test_input_name_is_input(self):
        """ONNX model input must be named 'input'."""
        from export_onnx import export_to_onnx

        model_path = tempfile.mktemp(suffix=".joblib")
        onnx_path = tempfile.mktemp(suffix=".onnx")

        try:
            _train_and_save_model(model_path)
            export_to_onnx(model_path, onnx_path)

            session = ort.InferenceSession(onnx_path)
            inputs = session.get_inputs()
            assert len(inputs) == 1
            assert inputs[0].name == "input"
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    def test_input_shape_is_none_16(self):
        """ONNX model input shape must be [None, 16]."""
        from export_onnx import export_to_onnx

        model_path = tempfile.mktemp(suffix=".joblib")
        onnx_path = tempfile.mktemp(suffix=".onnx")

        try:
            _train_and_save_model(model_path)
            export_to_onnx(model_path, onnx_path)

            session = ort.InferenceSession(onnx_path)
            input_shape = session.get_inputs()[0].shape
            # First dim is batch (None or dynamic), second is 16
            assert input_shape[1] == 16
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    def test_input_dtype_is_float32(self):
        """ONNX model input dtype must be float32."""
        from export_onnx import export_to_onnx

        model_path = tempfile.mktemp(suffix=".joblib")
        onnx_path = tempfile.mktemp(suffix=".onnx")

        try:
            _train_and_save_model(model_path)
            export_to_onnx(model_path, onnx_path)

            session = ort.InferenceSession(onnx_path)
            input_type = session.get_inputs()[0].type
            assert "float" in input_type.lower()
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    def test_output_name_is_probabilities(self):
        """ONNX model output must be named 'probabilities'."""
        from export_onnx import export_to_onnx

        model_path = tempfile.mktemp(suffix=".joblib")
        onnx_path = tempfile.mktemp(suffix=".onnx")

        try:
            _train_and_save_model(model_path)
            export_to_onnx(model_path, onnx_path)

            session = ort.InferenceSession(onnx_path)
            outputs = session.get_outputs()
            output_names = [o.name for o in outputs]
            assert "probabilities" in output_names
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    def test_output_shape_is_none_2(self):
        """ONNX model output shape must be [None, 2] (binary classification probs)."""
        from export_onnx import export_to_onnx

        model_path = tempfile.mktemp(suffix=".joblib")
        onnx_path = tempfile.mktemp(suffix=".onnx")

        try:
            _train_and_save_model(model_path)
            export_to_onnx(model_path, onnx_path)

            # Run inference and check actual output shape
            session = ort.InferenceSession(onnx_path)
            sample_input = np.zeros((1, 16), dtype=np.float32)
            results = session.run(
                ["probabilities"],
                {"input": sample_input},
            )
            probs = results[0]
            assert probs.shape == (1, 2), f"Expected (1, 2), got {probs.shape}"
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    def test_output_probabilities_sum_to_one(self):
        """Output probabilities should approximately sum to 1.0 for each sample."""
        from export_onnx import export_to_onnx

        model_path = tempfile.mktemp(suffix=".joblib")
        onnx_path = tempfile.mktemp(suffix=".onnx")

        try:
            _train_and_save_model(model_path)
            export_to_onnx(model_path, onnx_path)

            session = ort.InferenceSession(onnx_path)
            sample_input = np.random.rand(5, 16).astype(np.float32)
            results = session.run(
                ["probabilities"],
                {"input": sample_input},
            )
            probs = results[0]
            for i in range(5):
                assert probs[i].sum() == pytest.approx(1.0, abs=1e-4), (
                    f"Row {i} probs sum to {probs[i].sum()}, not 1.0"
                )
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    def test_output_dtype_is_float32(self):
        """Output probabilities must be float32."""
        from export_onnx import export_to_onnx

        model_path = tempfile.mktemp(suffix=".joblib")
        onnx_path = tempfile.mktemp(suffix=".onnx")

        try:
            _train_and_save_model(model_path)
            export_to_onnx(model_path, onnx_path)

            session = ort.InferenceSession(onnx_path)
            sample_input = np.zeros((1, 16), dtype=np.float32)
            results = session.run(
                ["probabilities"],
                {"input": sample_input},
            )
            probs = results[0]
            assert probs.dtype == np.float32
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    def test_batch_inference_works(self):
        """ONNX model should handle batch input correctly."""
        from export_onnx import export_to_onnx

        model_path = tempfile.mktemp(suffix=".joblib")
        onnx_path = tempfile.mktemp(suffix=".onnx")

        try:
            _train_and_save_model(model_path)
            export_to_onnx(model_path, onnx_path)

            session = ort.InferenceSession(onnx_path)
            batch_input = np.random.rand(10, 16).astype(np.float32)
            results = session.run(
                ["probabilities"],
                {"input": batch_input},
            )
            probs = results[0]
            assert probs.shape == (10, 2)
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)


# --- CLI tests ---


class TestExportOnnxCLI:
    """Test the export_onnx.py CLI behavior."""

    def test_help_flag_works(self):
        """--help should produce usage information and exit 0."""
        result = subprocess.run(
            [sys.executable, "export_onnx.py", "--help"],
            capture_output=True,
            text=True,
            cwd=TRAINING_DIR,
        )
        assert result.returncode == 0
        assert "export" in result.stdout.lower() or "onnx" in result.stdout.lower()

    def test_end_to_end_export(self):
        """Full export pipeline: train -> save joblib -> export ONNX -> validate."""
        model_path = tempfile.mktemp(suffix=".joblib")
        onnx_path = tempfile.mktemp(suffix=".onnx")

        try:
            _train_and_save_model(model_path)

            result = subprocess.run(
                [
                    sys.executable,
                    "export_onnx.py",
                    "--model",
                    model_path,
                    "--output",
                    onnx_path,
                ],
                capture_output=True,
                text=True,
                cwd=TRAINING_DIR,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            assert os.path.exists(onnx_path)

            # Validate the exported ONNX model
            session = ort.InferenceSession(onnx_path)
            assert session.get_inputs()[0].name == "input"

            sample = np.zeros((1, 16), dtype=np.float32)
            probs = session.run(["probabilities"], {"input": sample})[0]
            assert probs.shape == (1, 2)
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)
