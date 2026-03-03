"""Tests for split_dataset.py - dataset splitting functionality."""

import json
import os
import subprocess
import sys
import tempfile

import pytest

TRAINING_DIR = os.path.join(os.path.dirname(__file__), "..")


class TestSplitDatasetCLI:
    """Test the split_dataset.py CLI behavior."""

    def test_help_flag_works(self):
        """--help should produce usage information and exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "clt.split_dataset", "--help"],
            capture_output=True,
            text=True,
            cwd=TRAINING_DIR,
        )
        assert result.returncode == 0
        assert "Split dataset" in result.stdout

    def test_requires_input_argument(self):
        """Script should fail if --input is not provided."""
        result = subprocess.run(
            [sys.executable, "-m", "clt.split_dataset", "--output-dir", "/tmp/splits"],
            capture_output=True,
            text=True,
            cwd=TRAINING_DIR,
        )
        assert result.returncode != 0

    def test_requires_output_dir_argument(self):
        """Script should fail if --output-dir is not provided."""
        result = subprocess.run(
            [sys.executable, "-m", "clt.split_dataset", "--input", "/tmp/data.jsonl"],
            capture_output=True,
            text=True,
            cwd=TRAINING_DIR,
        )
        assert result.returncode != 0

    def test_excludes_uncertain_labels(self):
        """Samples with label=-1 should be excluded from all splits."""
        input_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        # Create 20 labeled samples and 5 uncertain
        for i in range(10):
            input_file.write(
                json.dumps({"file": f"a{i}.rs", "line": i, "label": 1}) + "\n"
            )
        for i in range(10):
            input_file.write(
                json.dumps({"file": f"b{i}.rs", "line": i, "label": 0}) + "\n"
            )
        for i in range(5):
            input_file.write(
                json.dumps({"file": f"c{i}.rs", "line": i, "label": -1}) + "\n"
            )
        input_file.close()

        output_dir = tempfile.mkdtemp()

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m", "clt.split_dataset",
                    "--input",
                    input_file.name,
                    "--output-dir",
                    output_dir,
                    "--seed",
                    "42",
                ],
                capture_output=True,
                text=True,
                cwd=TRAINING_DIR,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
            assert "5 uncertain excluded" in result.stdout

            # Verify no uncertain labels in any split
            total = 0
            for split_name in ["train", "val", "test"]:
                path = os.path.join(output_dir, f"{split_name}.jsonl")
                assert os.path.exists(path), f"{split_name}.jsonl should exist"
                with open(path) as f:
                    samples = [json.loads(line) for line in f if line.strip()]
                for s in samples:
                    assert s["label"] != -1, f"Uncertain label found in {split_name}"
                total += len(samples)

            # All 20 labeled samples should be distributed
            assert total == 20
        finally:
            os.unlink(input_file.name)
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_stratified_split_ratios(self):
        """Split should produce approximately 70/15/15 ratios."""
        input_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        # Create 100 samples per class for clearer ratio validation
        for i in range(100):
            input_file.write(
                json.dumps({"file": f"a{i}.rs", "line": i, "label": 1}) + "\n"
            )
        for i in range(100):
            input_file.write(
                json.dumps({"file": f"b{i}.rs", "line": i, "label": 0}) + "\n"
            )
        input_file.close()

        output_dir = tempfile.mkdtemp()

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m", "clt.split_dataset",
                    "--input",
                    input_file.name,
                    "--output-dir",
                    output_dir,
                    "--seed",
                    "42",
                ],
                capture_output=True,
                text=True,
                cwd=TRAINING_DIR,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"

            counts = {}
            for split_name in ["train", "val", "test"]:
                path = os.path.join(output_dir, f"{split_name}.jsonl")
                with open(path) as f:
                    samples = [json.loads(line) for line in f if line.strip()]
                counts[split_name] = len(samples)

            total = sum(counts.values())
            assert total == 200

            # Check approximate ratios (allow some rounding tolerance)
            assert 130 <= counts["train"] <= 150  # ~70% of 200 = 140
            assert 20 <= counts["val"] <= 40      # ~15% of 200 = 30
            assert 20 <= counts["test"] <= 40     # ~15% of 200 = 30
        finally:
            os.unlink(input_file.name)
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_creates_output_files(self):
        """Split should create train.jsonl, val.jsonl, and test.jsonl."""
        input_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        for i in range(10):
            input_file.write(
                json.dumps({"file": f"a{i}.rs", "line": i, "label": 1}) + "\n"
            )
        for i in range(10):
            input_file.write(
                json.dumps({"file": f"b{i}.rs", "line": i, "label": 0}) + "\n"
            )
        input_file.close()

        output_dir = tempfile.mkdtemp()

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m", "clt.split_dataset",
                    "--input",
                    input_file.name,
                    "--output-dir",
                    output_dir,
                ],
                capture_output=True,
                text=True,
                cwd=TRAINING_DIR,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"

            assert os.path.exists(os.path.join(output_dir, "train.jsonl"))
            assert os.path.exists(os.path.join(output_dir, "val.jsonl"))
            assert os.path.exists(os.path.join(output_dir, "test.jsonl"))
        finally:
            os.unlink(input_file.name)
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_deterministic_with_seed(self):
        """Same seed should produce same splits."""
        input_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        for i in range(50):
            input_file.write(
                json.dumps({"file": f"a{i}.rs", "line": i, "label": 1}) + "\n"
            )
        for i in range(50):
            input_file.write(
                json.dumps({"file": f"b{i}.rs", "line": i, "label": 0}) + "\n"
            )
        input_file.close()

        output_dir1 = tempfile.mkdtemp()
        output_dir2 = tempfile.mkdtemp()

        try:
            for output_dir in [output_dir1, output_dir2]:
                subprocess.run(
                    [
                        sys.executable,
                        "-m", "clt.split_dataset",
                        "--input",
                        input_file.name,
                        "--output-dir",
                        output_dir,
                        "--seed",
                        "42",
                    ],
                    capture_output=True,
                    text=True,
                    cwd=TRAINING_DIR,
                )

            for split_name in ["train", "val", "test"]:
                with open(os.path.join(output_dir1, f"{split_name}.jsonl")) as f:
                    samples1 = [json.loads(line) for line in f if line.strip()]
                with open(os.path.join(output_dir2, f"{split_name}.jsonl")) as f:
                    samples2 = [json.loads(line) for line in f if line.strip()]
                assert samples1 == samples2, f"{split_name} split differs between runs with same seed"
        finally:
            os.unlink(input_file.name)
            import shutil
            shutil.rmtree(output_dir1, ignore_errors=True)
            shutil.rmtree(output_dir2, ignore_errors=True)

    def test_class_balance_in_splits(self):
        """Each split should contain samples from both classes."""
        input_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        for i in range(50):
            input_file.write(
                json.dumps({"file": f"a{i}.rs", "line": i, "label": 1}) + "\n"
            )
        for i in range(50):
            input_file.write(
                json.dumps({"file": f"b{i}.rs", "line": i, "label": 0}) + "\n"
            )
        input_file.close()

        output_dir = tempfile.mkdtemp()

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m", "clt.split_dataset",
                    "--input",
                    input_file.name,
                    "--output-dir",
                    output_dir,
                    "--seed",
                    "42",
                ],
                capture_output=True,
                text=True,
                cwd=TRAINING_DIR,
            )
            assert result.returncode == 0

            for split_name in ["train", "val", "test"]:
                path = os.path.join(output_dir, f"{split_name}.jsonl")
                with open(path) as f:
                    samples = [json.loads(line) for line in f if line.strip()]
                labels = {s["label"] for s in samples}
                assert 0 in labels, f"{split_name} split missing label=0"
                assert 1 in labels, f"{split_name} split missing label=1"
        finally:
            os.unlink(input_file.name)
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_reports_class_distribution(self):
        """Output should report class distribution per split."""
        input_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        for i in range(20):
            input_file.write(
                json.dumps({"file": f"a{i}.rs", "line": i, "label": 1}) + "\n"
            )
        for i in range(20):
            input_file.write(
                json.dumps({"file": f"b{i}.rs", "line": i, "label": 0}) + "\n"
            )
        input_file.close()

        output_dir = tempfile.mkdtemp()

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m", "clt.split_dataset",
                    "--input",
                    input_file.name,
                    "--output-dir",
                    output_dir,
                ],
                capture_output=True,
                text=True,
                cwd=TRAINING_DIR,
            )
            assert result.returncode == 0
            assert "train:" in result.stdout
            assert "val:" in result.stdout
            assert "test:" in result.stdout
            assert "label=0:" in result.stdout
            assert "label=1:" in result.stdout
        finally:
            os.unlink(input_file.name)
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
