import pickle
from loguru import logger
from typing import Any


def save_pickle(obj: Any, filepath: str) -> None:
    """Save an object to disk using pickle."""
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved: {filepath}")


def load_pickle(filepath: str) -> Any:
    """Load a pickled object from disk."""
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Loaded: {filepath}")
    return obj
