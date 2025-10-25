"""Unit tests for utils module."""

from unittest.mock import patch

import pytest

from src.modelling.utils import load_pickle, save_pickle

pytestmark = pytest.mark.unit


class TestSavePickle:
    """Test cases for save_pickle function."""

    def test_save_pickle_saves_object(self, tmp_path):
        """Test that save_pickle saves object correctly."""
        dummy_obj = {"a": 1, "b": 2, "c": [1, 2, 3]}
        filepath = tmp_path / "test.pkl"

        save_pickle(dummy_obj, str(filepath))

        assert filepath.exists()
        assert filepath.stat().st_size > 0

    def test_save_pickle_with_nested_object(self, tmp_path):
        """Test save_pickle with nested objects."""
        nested_obj = {"data": {"nested": [1, 2, 3]}, "metadata": {"version": "1.0"}}
        filepath = tmp_path / "nested.pkl"

        save_pickle(nested_obj, str(filepath))

        assert filepath.exists()

    def test_save_pickle_creates_directory(self, tmp_path):
        """Test that save_pickle creates directory if it doesn't exist."""
        dummy_obj = {"test": "data"}
        filepath = tmp_path / "new_dir" / "test.pkl"

        save_pickle(dummy_obj, str(filepath))

        assert filepath.exists()
        assert filepath.parent.exists()

    @patch("src.modelling.utils.logger")
    def test_save_pickle_logs_success(self, mock_logger, tmp_path):
        """Test that save_pickle logs success message."""
        dummy_obj = {"test": "data"}
        filepath = tmp_path / "test.pkl"

        save_pickle(dummy_obj, str(filepath))

        mock_logger.info.assert_called()


class TestLoadPickle:
    """Test cases for load_pickle function."""

    def test_load_pickle_loads_object(self, tmp_path):
        """Test that load_pickle loads object correctly."""
        dummy_obj = {"a": 1, "b": 2, "c": [1, 2, 3]}
        filepath = tmp_path / "test.pkl"

        # Save first
        save_pickle(dummy_obj, str(filepath))

        # Then load
        loaded_obj = load_pickle(str(filepath))

        assert loaded_obj == dummy_obj

    def test_load_pickle_with_nested_object(self, tmp_path):
        """Test load_pickle with nested objects."""
        nested_obj = {"data": {"nested": [1, 2, 3]}, "metadata": {"version": "1.0"}}
        filepath = tmp_path / "nested.pkl"

        save_pickle(nested_obj, str(filepath))
        loaded_obj = load_pickle(str(filepath))

        assert loaded_obj == nested_obj

    def test_load_pickle_file_not_found(self, tmp_path):
        """Test load_pickle with non-existent file."""
        filepath = tmp_path / "nonexistent.pkl"

        with pytest.raises(FileNotFoundError):
            load_pickle(str(filepath))

    def test_load_pickle_corrupted_file(self, tmp_path):
        """Test load_pickle with corrupted file."""
        filepath = tmp_path / "corrupted.pkl"
        filepath.write_text("corrupted data")

        with pytest.raises(Exception):  # Should raise some kind of pickle error
            load_pickle(str(filepath))

    @patch("src.modelling.utils.logger")
    def test_load_pickle_logs_success(self, mock_logger, tmp_path):
        """Test that load_pickle logs success message."""
        dummy_obj = {"test": "data"}
        filepath = tmp_path / "test.pkl"

        save_pickle(dummy_obj, str(filepath))
        load_pickle(str(filepath))

        mock_logger.info.assert_called()


class TestSaveAndLoadIntegration:
    """Integration tests for save_pickle and load_pickle."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Test complete save and load cycle."""
        original_obj = {
            "string": "test",
            "number": 42,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "boolean": True,
            "none": None,
        }
        filepath = tmp_path / "roundtrip.pkl"

        # Save
        save_pickle(original_obj, str(filepath))

        # Load
        loaded_obj = load_pickle(str(filepath))

        assert loaded_obj == original_obj
        assert loaded_obj["string"] == "test"
        assert loaded_obj["number"] == 42
        assert loaded_obj["list"] == [1, 2, 3]
        assert loaded_obj["dict"] == {"nested": "value"}
        assert loaded_obj["boolean"] is True
        assert loaded_obj["none"] is None
