"""Unit tests for config module."""

from pathlib import Path

import pytest

from src.modelling import config

pytestmark = pytest.mark.unit


class TestConfig:
    """Test cases for config module."""

    def test_config_constants(self):
        """Test that all config constants are properly defined."""
        assert config.MODEL_TYPE == "RandomForest"
        assert config.MODEL_VERSION == "0.1.0"
        assert config.TARGET == "Age"
        assert config.TEST_SIZE == 0.2
        assert config.RANDOM_STATE == 42

    def test_config_paths(self):
        """Test that config paths are properly defined."""
        assert isinstance(config.PROJECT_ROOT, Path)
        assert isinstance(config.DATA_DIR, Path)
        assert isinstance(config.MODELS_DIR, Path)
        assert isinstance(config.PIPELINE_PATH, Path)

        # Check that paths are correctly constructed
        assert config.DATA_DIR.name == "data"
        assert config.MODELS_DIR.name == "models"
        assert config.PIPELINE_PATH.name == "pipeline__v0.1.0.pkl"

    def test_categorical_columns(self):
        """Test categorical columns configuration."""
        assert config.CATEGORICAL_COLS == ["Sex"]
        assert len(config.CATEGORICAL_COLS) == 1
        assert "Sex" in config.CATEGORICAL_COLS

    def test_numeric_columns(self):
        """Test numeric columns configuration."""
        expected_numeric_cols = [
            "Length",
            "Diameter",
            "Height",
            "Whole weight",
            "Shucked weight",
            "Viscera weight",
            "Shell weight",
        ]
        assert config.NUMERIC_COLS == expected_numeric_cols
        assert len(config.NUMERIC_COLS) == 7
        assert all(col in config.NUMERIC_COLS for col in expected_numeric_cols)

    def test_models_dir_creation(self):
        """Test that MODELS_DIR is created."""
        assert config.MODELS_DIR.exists()
