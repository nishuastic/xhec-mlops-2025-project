"""Integration tests for training pipeline."""

import pandas as pd
import pytest

from src.modelling.config import MODEL_TYPE
from src.modelling.preprocessing import prepare_features
from src.modelling.training import evaluate_model, train_model

pytestmark = pytest.mark.integration


class TestTrainingIntegration:
    """Integration tests for the complete training pipeline."""

    def test_full_training_pipeline(self, sample_dataframe):
        """Test the complete training pipeline end-to-end."""
        # Prepare features
        X_train, X_test, y_train, y_test, preprocessor = prepare_features(
            sample_dataframe
        )

        # Train model
        pipe = train_model(X_train, y_train, preprocessor, model_type=MODEL_TYPE)

        # Evaluate model
        metrics = evaluate_model(pipe, X_test, y_test)

        # Assertions
        assert "RMSE" in metrics
        assert "MAE" in metrics
        assert "R2" in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())
        assert hasattr(pipe, "predict")  # Pipeline should have predict method

    def test_training_with_different_model_types(self, sample_dataframe):
        """Test training pipeline with different model types."""
        model_types = ["LinearRegression", "Ridge", "RandomForest", "GradientBoosting"]

        for model_type in model_types:
            X_train, X_test, y_train, y_test, preprocessor = prepare_features(
                sample_dataframe
            )
            pipe = train_model(X_train, y_train, preprocessor, model_type=model_type)
            metrics = evaluate_model(pipe, X_test, y_test)

            assert "RMSE" in metrics
            assert "MAE" in metrics
            assert "R2" in metrics

    def test_training_pipeline_consistency(self, sample_dataframe):
        """Test that training pipeline produces consistent results."""
        # Run the same pipeline twice
        X_train, X_test, y_train, y_test, preprocessor = prepare_features(
            sample_dataframe
        )

        pipe1 = train_model(X_train, y_train, preprocessor, model_type="RandomForest")
        metrics1 = evaluate_model(pipe1, X_test, y_test)

        pipe2 = train_model(X_train, y_train, preprocessor, model_type="RandomForest")
        metrics2 = evaluate_model(pipe2, X_test, y_test)

        # Results should be identical due to fixed random state
        assert metrics1["RMSE"] == metrics2["RMSE"]
        assert metrics1["MAE"] == metrics2["MAE"]
        assert metrics1["R2"] == metrics2["R2"]

    def test_training_with_minimal_data(self):
        """Test training pipeline with minimal data."""
        # Create minimal dataset
        df = pd.DataFrame(
            {
                "Sex": ["M", "F", "I", "M", "F", "I", "M", "F"],
                "Length": [0.5, 0.6, 0.4, 0.55, 0.52, 0.48, 0.58, 0.45],
                "Diameter": [0.4, 0.45, 0.35, 0.42, 0.38, 0.33, 0.46, 0.36],
                "Height": [0.1, 0.15, 0.08, 0.12, 0.11, 0.09, 0.14, 0.10],
                "Whole weight": [0.2, 0.25, 0.15, 0.22, 0.21, 0.18, 0.24, 0.19],
                "Shucked weight": [0.1, 0.12, 0.09, 0.11, 0.10, 0.08, 0.13, 0.09],
                "Viscera weight": [0.05, 0.06, 0.04, 0.05, 0.04, 0.03, 0.07, 0.04],
                "Shell weight": [0.07, 0.08, 0.05, 0.07, 0.06, 0.04, 0.09, 0.05],
                "Age": [6.5, 11.5, 4.5, 7.5, 6.8, 5.2, 8.1, 5.9],
            }
        )

        X_train, X_test, y_train, y_test, preprocessor = prepare_features(df)
        pipe = train_model(
            X_train, y_train, preprocessor, model_type="LinearRegression"
        )
        metrics = evaluate_model(pipe, X_test, y_test)

        assert "RMSE" in metrics
        assert "MAE" in metrics
        assert "R2" in metrics

    def test_training_pipeline_error_handling(self):
        """Test training pipeline with invalid data."""
        # Create invalid dataset (missing required columns)
        df = pd.DataFrame(
            {
                "Sex": ["M", "F"],
                "Length": [0.5, 0.6],
                # Missing other required columns
            }
        )

        with pytest.raises((KeyError, ValueError)):
            prepare_features(df)
