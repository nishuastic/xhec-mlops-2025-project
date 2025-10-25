"""Unit tests for training module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.modelling.training import evaluate_model, train_model

pytestmark = pytest.mark.unit


class TestTrainModel:
    """Test cases for train_model function."""

    def test_train_model_creates_pipeline(self, sample_dataframe, mock_preprocessor):
        """Test that train_model creates a pipeline."""
        X_train = sample_dataframe.drop(columns=["Age"])
        y_train = sample_dataframe["Age"]

        pipe = train_model(
            X_train, y_train, mock_preprocessor, model_type="LinearRegression"
        )

        assert isinstance(pipe, Pipeline)
        assert "preprocessor" in pipe.named_steps
        assert "model" in pipe.named_steps

    def test_train_model_with_different_model_types(
        self, sample_dataframe, mock_preprocessor
    ):
        """Test train_model with different model types."""
        X_train = sample_dataframe.drop(columns=["Age"])
        y_train = sample_dataframe["Age"]

        model_types = ["LinearRegression", "Ridge", "RandomForest", "GradientBoosting"]

        for model_type in model_types:
            pipe = train_model(
                X_train, y_train, mock_preprocessor, model_type=model_type
            )
            assert isinstance(pipe, Pipeline)

    def test_train_model_with_invalid_model_type(
        self, sample_dataframe, mock_preprocessor
    ):
        """Test train_model with invalid model type raises ValueError."""
        X_train = sample_dataframe.drop(columns=["Age"])
        y_train = sample_dataframe["Age"]

        with pytest.raises(ValueError, match="Unknown model type"):
            train_model(X_train, y_train, mock_preprocessor, model_type="InvalidModel")

    @patch("src.modelling.training.logger")
    def test_train_model_logs_success(
        self, mock_logger, sample_dataframe, mock_preprocessor
    ):
        """Test that train_model logs success message."""
        X_train = sample_dataframe.drop(columns=["Age"])
        y_train = sample_dataframe["Age"]

        train_model(X_train, y_train, mock_preprocessor, model_type="LinearRegression")

        mock_logger.success.assert_called()


class TestEvaluateModel:
    """Test cases for evaluate_model function."""

    def test_evaluate_model_returns_metrics(self, mock_model):
        """Test that evaluate_model returns expected metrics."""
        # Create a mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = np.array([1.0, 2.0, 3.0])

        X_test = pd.DataFrame({"feature": [1, 2, 3]})
        y_test = pd.Series([1.1, 2.1, 3.1])

        metrics = evaluate_model(mock_pipeline, X_test, y_test)

        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "R2" in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())

    def test_evaluate_model_calls_predict(self, mock_model):
        """Test that evaluate_model calls pipeline.predict."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = np.array([1.0, 2.0, 3.0])

        X_test = pd.DataFrame({"feature": [1, 2, 3]})
        y_test = pd.Series([1.1, 2.1, 3.1])

        evaluate_model(mock_pipeline, X_test, y_test)

        mock_pipeline.predict.assert_called_once_with(X_test)

    @patch("src.modelling.training.logger")
    def test_evaluate_model_logs_metrics(self, mock_logger, mock_model):
        """Test that evaluate_model logs metrics."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = np.array([1.0, 2.0, 3.0])

        X_test = pd.DataFrame({"feature": [1, 2, 3]})
        y_test = pd.Series([1.1, 2.1, 3.1])

        evaluate_model(mock_pipeline, X_test, y_test)

        mock_logger.info.assert_called()

    def test_evaluate_model_with_perfect_predictions(self):
        """Test evaluate_model with perfect predictions."""
        mock_pipeline = MagicMock()
        y_test = pd.Series([1.0, 2.0, 3.0])
        mock_pipeline.predict.return_value = y_test.values

        X_test = pd.DataFrame({"feature": [1, 2, 3]})

        metrics = evaluate_model(mock_pipeline, X_test, y_test)

        assert metrics["R2"] == 1.0  # Perfect R2 score
        assert metrics["MAE"] == 0.0  # Perfect MAE
        assert metrics["RMSE"] == 0.0  # Perfect RMSE

    def test_evaluate_model_with_empty_predictions(self):
        """Test evaluate_model with empty predictions."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = np.array([])

        X_test = pd.DataFrame()
        y_test = pd.Series([], dtype=float)

        with pytest.raises(ValueError):
            evaluate_model(mock_pipeline, X_test, y_test)
