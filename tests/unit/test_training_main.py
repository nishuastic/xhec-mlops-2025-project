"""Unit tests for training module main execution."""

import pytest

pytestmark = pytest.mark.unit


class TestTrainingMain:
    """Test cases for training module main execution."""

    def test_training_main_execution(self):
        """Test training module main execution."""
        # The training module doesn't have a main execution block
        # but we can test that it can be imported and has the expected structure
        import src.modelling.training

        # Verify the module can be imported
        assert src.modelling.training is not None

        # Verify the module has the expected functions
        assert hasattr(src.modelling.training, "train_model")
        assert hasattr(src.modelling.training, "evaluate_model")
        assert hasattr(src.modelling.training, "save_artifacts")

    def test_training_module_imports(self):
        """Test that all required imports are available."""
        from src.modelling.training import evaluate_model, save_artifacts, train_model

        # Verify all functions are callable
        assert callable(train_model)
        assert callable(evaluate_model)
        assert callable(save_artifacts)

    def test_training_module_structure(self):
        """Test the overall structure of the training module."""
        import src.modelling.training

        # Verify the module has the expected attributes
        assert hasattr(src.modelling.training, "train_model")
        assert hasattr(src.modelling.training, "evaluate_model")
        assert hasattr(src.modelling.training, "save_artifacts")

        # Verify functions are callable
        from src.modelling.training import evaluate_model, save_artifacts, train_model

        assert callable(train_model)
        assert callable(evaluate_model)
        assert callable(save_artifacts)
