import pickle
from pathlib import Path
from typing import Any

from loguru import logger


def save_pickle(obj: Any, filepath: str) -> None:
    """Save an object to disk using pickle."""
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved: {filepath}")


def load_pickle(filepath: str) -> Any:
    """Load a pickled object from disk."""
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Loaded: {filepath}")
    return obj
