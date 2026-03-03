"""Tests for label_review.py - interactive label review tool."""

import json
import os
import subprocess
import sys
import tempfile

import pytest


class TestDisplaySample:
    """Test the display_sample function output formatting."""

    def test_display_sample_shows_file_info(self, capsys):
        """display_sample should show file path and line number."""
        from label_review import display_sample

        sample = {
            "file": "src/main.rs",
            "line": 42,
            "language": "rust",
            "comment_text": "// increment counter",
            "heuristic_score": 0.85,
            "heuristic_confidence": 0.9,
            "features": {
                "token_overlap_jaccard": 0.91,
                "identifier_substring_ratio": 0.5,
                "has_why_indicator": False,
                "has_external_ref": False,
                "imperative_verb_noun": False,
                "is_doc_comment": False,
            },
        }

        display_sample(sample, 0, 1)
        captured = capsys.readouterr()

        assert "src/main.rs:42" in captured.out
        assert "rust" in captured.out
        assert "increment counter" in captured.out
        assert "0.850" in captured.out
        assert "0.910" in captured.out

    def test_display_sample_handles_missing_features(self, capsys):
        """display_sample should handle samples with no features dict."""
        from label_review import display_sample

        sample = {
            "file": "test.py",
            "line": 1,
            "language": "python",
            "comment_text": "# test",
            "heuristic_score": 0.5,
            "heuristic_confidence": 0.5,
            "features": {},
        }

        display_sample(sample, 0, 1)
        captured = capsys.readouterr()

        assert "test.py:1" in captured.out
        assert "0.000" in captured.out  # default feature value


class TestLabelReviewCLI:
    """Test the label_review.py CLI behavior."""

    def test_help_flag_works(self):
        """--help should produce usage information and exit 0."""
        result = subprocess.run(
            [sys.executable, "label_review.py", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        assert result.returncode == 0
        assert "Review uncertain labels" in result.stdout

    def test_requires_input_argument(self):
        """Script should fail if --input is not provided."""
        result = subprocess.run(
            [sys.executable, "label_review.py", "--output", "/tmp/out.jsonl"],
            capture_output=True,
            text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        assert result.returncode != 0

    def test_requires_output_argument(self):
        """Script should fail if --output is not provided."""
        result = subprocess.run(
            [sys.executable, "label_review.py", "--input", "/tmp/in.jsonl"],
            capture_output=True,
            text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        assert result.returncode != 0

    def test_loads_and_counts_uncertain_samples(self):
        """Script should correctly identify uncertain samples from input."""
        # Create input with mix of labels
        input_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        samples = [
            {"file": "a.rs", "line": 1, "label": 1, "label_source": "heuristic_auto",
             "comment_text": "x", "heuristic_score": 0.9, "heuristic_confidence": 0.9,
             "language": "rust", "features": {}},
            {"file": "b.rs", "line": 2, "label": -1, "label_source": "uncertain",
             "comment_text": "y", "heuristic_score": 0.5, "heuristic_confidence": 0.5,
             "language": "rust", "features": {}},
            {"file": "c.rs", "line": 3, "label": 0, "label_source": "heuristic_auto",
             "comment_text": "z", "heuristic_score": 0.1, "heuristic_confidence": 0.9,
             "language": "rust", "features": {}},
        ]
        for s in samples:
            input_file.write(json.dumps(s) + "\n")
        input_file.close()

        output_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        output_file.close()

        try:
            # Simulate immediate quit
            result = subprocess.run(
                [
                    sys.executable,
                    "label_review.py",
                    "--input",
                    input_file.name,
                    "--output",
                    output_file.name,
                ],
                input="q\n",
                capture_output=True,
                text=True,
                cwd=os.path.join(os.path.dirname(__file__), ".."),
            )
            assert result.returncode == 0
            assert "1 uncertain samples" in result.stdout

            # Output should contain all 3 samples (unchanged since we quit)
            with open(output_file.name) as f:
                output_samples = [json.loads(line) for line in f if line.strip()]
            assert len(output_samples) == 3
        finally:
            os.unlink(input_file.name)
            os.unlink(output_file.name)

    def test_label_superfluous_choice(self):
        """Choosing 's' should set label to 1 and source to manual_review."""
        input_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        sample = {
            "file": "a.rs", "line": 1, "label": -1, "label_source": "uncertain",
            "comment_text": "// test", "heuristic_score": 0.5,
            "heuristic_confidence": 0.5, "language": "rust", "features": {},
        }
        input_file.write(json.dumps(sample) + "\n")
        input_file.close()

        output_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        output_file.close()

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "label_review.py",
                    "--input",
                    input_file.name,
                    "--output",
                    output_file.name,
                ],
                input="s\n",
                capture_output=True,
                text=True,
                cwd=os.path.join(os.path.dirname(__file__), ".."),
            )
            assert result.returncode == 0

            with open(output_file.name) as f:
                output_samples = [json.loads(line) for line in f if line.strip()]

            assert len(output_samples) == 1
            assert output_samples[0]["label"] == 1
            assert output_samples[0]["label_source"] == "manual_review"
        finally:
            os.unlink(input_file.name)
            os.unlink(output_file.name)

    def test_label_valuable_choice(self):
        """Choosing 'v' should set label to 0 and source to manual_review."""
        input_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        sample = {
            "file": "a.rs", "line": 1, "label": -1, "label_source": "uncertain",
            "comment_text": "// test", "heuristic_score": 0.5,
            "heuristic_confidence": 0.5, "language": "rust", "features": {},
        }
        input_file.write(json.dumps(sample) + "\n")
        input_file.close()

        output_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        output_file.close()

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "label_review.py",
                    "--input",
                    input_file.name,
                    "--output",
                    output_file.name,
                ],
                input="v\n",
                capture_output=True,
                text=True,
                cwd=os.path.join(os.path.dirname(__file__), ".."),
            )
            assert result.returncode == 0

            with open(output_file.name) as f:
                output_samples = [json.loads(line) for line in f if line.strip()]

            assert len(output_samples) == 1
            assert output_samples[0]["label"] == 0
            assert output_samples[0]["label_source"] == "manual_review"
        finally:
            os.unlink(input_file.name)
            os.unlink(output_file.name)
