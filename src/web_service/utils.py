import pickle
from typing import Any

from loguru import logger


def load_pickle(filepath: str) -> Any:
    """Load a pickled object from disk."""
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Loaded: {filepath}")
    return obj
