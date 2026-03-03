"""Tests for generate_data.py - labeled training data generation."""

import json
import os
import subprocess
import sys
import tempfile

import pytest


# --- Tests for label_from_score ---


class TestLabelFromScore:
    """Test the heuristic score to label mapping function."""

    def test_high_score_is_superfluous(self):
        """Score >= 0.7 should be labeled as superfluous (label=1)."""
        from generate_data import label_from_score

        label, source = label_from_score(0.7)
        assert label == 1
        assert source == "heuristic_auto"

    def test_very_high_score_is_superfluous(self):
        """Score of 0.95 should be labeled as superfluous."""
        from generate_data import label_from_score

        label, source = label_from_score(0.95)
        assert label == 1
        assert source == "heuristic_auto"

    def test_score_exactly_one_is_superfluous(self):
        """Score of 1.0 should be labeled as superfluous."""
        from generate_data import label_from_score

        label, source = label_from_score(1.0)
        assert label == 1
        assert source == "heuristic_auto"

    def test_low_score_is_valuable(self):
        """Score <= 0.3 should be labeled as valuable (label=0)."""
        from generate_data import label_from_score

        label, source = label_from_score(0.3)
        assert label == 0
        assert source == "heuristic_auto"

    def test_very_low_score_is_valuable(self):
        """Score of 0.1 should be labeled as valuable."""
        from generate_data import label_from_score

        label, source = label_from_score(0.1)
        assert label == 0
        assert source == "heuristic_auto"

    def test_zero_score_is_valuable(self):
        """Score of 0.0 should be labeled as valuable."""
        from generate_data import label_from_score

        label, source = label_from_score(0.0)
        assert label == 0
        assert source == "heuristic_auto"

    def test_mid_score_is_uncertain(self):
        """Score between 0.3 and 0.7 (exclusive) should be uncertain (label=-1)."""
        from generate_data import label_from_score

        label, source = label_from_score(0.5)
        assert label == -1
        assert source == "uncertain"

    def test_just_above_low_threshold_is_uncertain(self):
        """Score just above 0.3 should be uncertain."""
        from generate_data import label_from_score

        label, source = label_from_score(0.31)
        assert label == -1
        assert source == "uncertain"

    def test_just_below_high_threshold_is_uncertain(self):
        """Score just below 0.7 should be uncertain."""
        from generate_data import label_from_score

        label, source = label_from_score(0.69)
        assert label == -1
        assert source == "uncertain"


# --- Tests for CLI integration ---


class TestGenerateDataCLI:
    """Test the generate_data.py CLI behavior."""

    def test_help_flag_works(self):
        """--help should produce usage information and exit 0."""
        result = subprocess.run(
            [sys.executable, "generate_data.py", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        assert result.returncode == 0
        assert "Generate labeled training data" in result.stdout

    def test_requires_binary_argument(self):
        """Script should fail if --binary is not provided."""
        result = subprocess.run(
            [sys.executable, "generate_data.py", "--dir", "/tmp"],
            capture_output=True,
            text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        assert result.returncode != 0

    def test_requires_dir_argument(self):
        """Script should fail if --dir is not provided."""
        result = subprocess.run(
            [sys.executable, "generate_data.py", "--binary", "/usr/bin/echo"],
            capture_output=True,
            text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        assert result.returncode != 0

    def test_output_to_file(self):
        """Script should write labeled JSONL to --output file when binary outputs valid JSONL."""
        # Create a mock binary that outputs JSONL to stdout
        mock_binary = tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False
        )
        mock_binary.write("#!/bin/bash\n")
        mock_binary.write(
            'echo \'{"file":"test.rs","line":1,"column":0,"language":"rust",'
            '"comment_text":"// increment counter","comment_kind":"line",'
            '"heuristic_score":0.85,"heuristic_confidence":0.9,"features":{}}\'\n'
        )
        mock_binary.write(
            'echo \'{"file":"test.rs","line":2,"column":0,"language":"rust",'
            '"comment_text":"// important safety note","comment_kind":"line",'
            '"heuristic_score":0.2,"heuristic_confidence":0.8,"features":{}}\'\n'
        )
        mock_binary.write(
            'echo \'{"file":"test.rs","line":3,"column":0,"language":"rust",'
            '"comment_text":"// maybe useful","comment_kind":"line",'
            '"heuristic_score":0.5,"heuristic_confidence":0.6,"features":{}}\'\n'
        )
        mock_binary.close()
        os.chmod(mock_binary.name, 0o755)

        output_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        output_file.close()

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "generate_data.py",
                    "--binary",
                    mock_binary.name,
                    "--dir",
                    "/tmp",
                    "--output",
                    output_file.name,
                ],
                capture_output=True,
                text=True,
                cwd=os.path.join(os.path.dirname(__file__), ".."),
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"

            # Read output and verify labels
            with open(output_file.name) as f:
                lines = [json.loads(line) for line in f if line.strip()]

            assert len(lines) == 3

            # High score -> superfluous
            assert lines[0]["label"] == 1
            assert lines[0]["label_source"] == "heuristic_auto"

            # Low score -> valuable
            assert lines[1]["label"] == 0
            assert lines[1]["label_source"] == "heuristic_auto"

            # Mid score -> uncertain
            assert lines[2]["label"] == -1
            assert lines[2]["label_source"] == "uncertain"
        finally:
            os.unlink(mock_binary.name)
            os.unlink(output_file.name)

    def test_stderr_reports_counts(self):
        """Script should report counts to stderr."""
        mock_binary = tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False
        )
        mock_binary.write("#!/bin/bash\n")
        mock_binary.write(
            'echo \'{"file":"t.rs","line":1,"column":0,"language":"rust",'
            '"comment_text":"x","comment_kind":"line",'
            '"heuristic_score":0.85,"heuristic_confidence":0.9,"features":{}}\'\n'
        )
        mock_binary.close()
        os.chmod(mock_binary.name, 0o755)

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "generate_data.py",
                    "--binary",
                    mock_binary.name,
                    "--dir",
                    "/tmp",
                ],
                capture_output=True,
                text=True,
                cwd=os.path.join(os.path.dirname(__file__), ".."),
            )
            assert result.returncode == 0
            assert "Generated" in result.stderr
            assert "Superfluous" in result.stderr
        finally:
            os.unlink(mock_binary.name)
