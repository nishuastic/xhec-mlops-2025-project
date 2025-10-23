from src.modelling.utils import save_pickle, load_pickle
import tempfile
import os


def test_save_and_load_pickle():
    """Check saving and loading a pickle file."""
    dummy_obj = {"a": 1, "b": 2}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pkl")
        save_pickle(dummy_obj, path)
        loaded = load_pickle(path)

    assert loaded == dummy_obj
