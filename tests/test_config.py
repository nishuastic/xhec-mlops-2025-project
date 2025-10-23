from src.modelling import config
from pathlib import Path


def test_config_paths_exist():
    """Check that config paths and constants exist."""
    assert isinstance(config.DATA_DIR, Path)
    assert isinstance(config.MODELS_DIR, Path)
    assert config.MODELS_DIR.exists()
    assert config.TARGET == "Age"
    assert "Sex" in config.CATEGORICAL_COLS
    assert "Length" in config.NUMERIC_COLS
