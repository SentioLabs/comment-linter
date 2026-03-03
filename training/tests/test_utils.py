"""Tests for utils.py - shared training pipeline utilities."""

import json
import os
import tempfile

import pytest

from clt.utils import (
    BOOL_FEATURES,
    FEATURE_DIM,
    FEATURE_NAMES,
    OUTPUT_DIM,
    load_jsonl,
    to_bool,
    to_float,
    to_int,
    write_jsonl,
)


class TestConstants:
    """Verify feature schema constants match tensor.rs contract."""

    def test_feature_dim_is_16(self):
        assert FEATURE_DIM == 16

    def test_output_dim_is_2(self):
        assert OUTPUT_DIM == 2

    def test_feature_names_length_matches_dim(self):
        assert len(FEATURE_NAMES) == FEATURE_DIM

    def test_bool_features_count_is_11(self):
        assert len(BOOL_FEATURES) == 11

    def test_bool_features_are_subset_of_feature_names(self):
        assert BOOL_FEATURES <= set(FEATURE_NAMES)

    def test_feature_names_first_and_last(self):
        assert FEATURE_NAMES[0] == "token_overlap_jaccard"
        assert FEATURE_NAMES[-1] == "comment_code_age_ratio"


class TestJsonlRoundTrip:
    """Test JSONL read/write round-trip."""

    def test_write_then_load(self):
        records = [{"a": 1, "b": "hello"}, {"a": 2, "b": "world"}]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name

        try:
            write_jsonl(records, path)
            loaded = load_jsonl(path)
            assert loaded == records
        finally:
            os.unlink(path)

    def test_load_skips_empty_lines(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            f.write('{"x": 1}\n')
            f.write("\n")
            f.write('{"x": 2}\n')
            f.write("   \n")
            path = f.name

        try:
            loaded = load_jsonl(path)
            assert len(loaded) == 2
        finally:
            os.unlink(path)

    def test_empty_file_returns_empty_list(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name

        try:
            loaded = load_jsonl(path)
            assert loaded == []
        finally:
            os.unlink(path)

    def test_accepts_path_object(self):
        from pathlib import Path

        records = [{"val": 42}]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            path = Path(f.name)

        try:
            write_jsonl(records, path)
            loaded = load_jsonl(path)
            assert loaded == records
        finally:
            os.unlink(str(path))


class TestToBool:
    """Test bool coercion edge cases."""

    def test_bool_passthrough(self):
        assert to_bool(True) is True
        assert to_bool(False) is False

    def test_int_nonzero(self):
        assert to_bool(1) is True
        assert to_bool(0) is False
        assert to_bool(-1) is True

    def test_float_nonzero(self):
        assert to_bool(1.0) is True
        assert to_bool(0.0) is False

    def test_string_truthy(self):
        assert to_bool("true") is True
        assert to_bool("True") is True
        assert to_bool("1") is True
        assert to_bool("yes") is True
        assert to_bool("y") is True

    def test_string_falsy(self):
        assert to_bool("false") is False
        assert to_bool("0") is False
        assert to_bool("no") is False
        assert to_bool("") is False

    def test_none_returns_false(self):
        assert to_bool(None) is False


class TestToInt:
    """Test int coercion edge cases."""

    def test_int_passthrough(self):
        assert to_int(42) == 42

    def test_float_truncates(self):
        assert to_int(3.7) == 3

    def test_string_numeric(self):
        assert to_int("5") == 5

    def test_invalid_returns_default(self):
        assert to_int("abc") == 0
        assert to_int("abc", default=-1) == -1

    def test_none_returns_default(self):
        assert to_int(None) == 0
        assert to_int(None, default=99) == 99


class TestToFloat:
    """Test float coercion edge cases."""

    def test_float_passthrough(self):
        assert to_float(3.14) == pytest.approx(3.14)

    def test_int_converts(self):
        assert to_float(5) == pytest.approx(5.0)

    def test_string_numeric(self):
        assert to_float("2.5") == pytest.approx(2.5)

    def test_invalid_returns_default(self):
        assert to_float("abc") == pytest.approx(0.0)
        assert to_float("abc", default=-1.0) == pytest.approx(-1.0)

    def test_none_returns_default(self):
        assert to_float(None) == pytest.approx(0.0)
