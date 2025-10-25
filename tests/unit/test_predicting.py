"""Unit tests for predicting module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.modelling.predicting import predict

pytestmark = pytest.mark.unit


class TestPredicting:
    """Test cases for predicting module."""

    @patch("src.modelling.predicting.load_pickle")
    @patch("src.modelling.predicting.logger")
    def test_predict_success(self, mock_logger, mock_load_pickle):
        """Test successful prediction."""
        # Setup mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = [5.2, 7.8, 3.1]
        mock_load_pickle.return_value = mock_pipeline

        # Test data
        test_data = pd.DataFrame(
            {
                "Sex": ["M", "F", "I"],
                "Length": [0.5, 0.6, 0.4],
                "Diameter": [0.4, 0.45, 0.35],
                "Height": [0.1, 0.15, 0.08],
                "Whole weight": [0.2, 0.25, 0.15],
                "Shucked weight": [0.1, 0.12, 0.09],
                "Viscera weight": [0.05, 0.06, 0.04],
                "Shell weight": [0.07, 0.08, 0.05],
            }
        )

        # Call function
        result = predict(test_data)

        # Assertions
        assert result == [5.2, 7.8, 3.1]
        mock_load_pickle.assert_called_once()
        mock_pipeline.predict.assert_called_once_with(test_data)
        mock_logger.success.assert_called_once_with("Prediction complete.")

    @patch("src.modelling.predicting.load_pickle")
    def test_predict_with_single_row(self, mock_load_pickle):
        """Test prediction with single row of data."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = [5.2]
        mock_load_pickle.return_value = mock_pipeline

        test_data = pd.DataFrame(
            {
                "Sex": ["M"],
                "Length": [0.5],
                "Diameter": [0.4],
                "Height": [0.1],
                "Whole weight": [0.2],
                "Shucked weight": [0.1],
                "Viscera weight": [0.05],
                "Shell weight": [0.07],
            }
        )

        result = predict(test_data)

        assert result == [5.2]
        mock_pipeline.predict.assert_called_once_with(test_data)

    @patch("src.modelling.predicting.load_pickle")
    def test_predict_with_empty_dataframe(self, mock_load_pickle):
        """Test prediction with empty dataframe."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = []
        mock_load_pickle.return_value = mock_pipeline

        test_data = pd.DataFrame()

        result = predict(test_data)

        assert result == []
        mock_pipeline.predict.assert_called_once_with(test_data)

    @patch("src.modelling.predicting.load_pickle")
    def test_predict_pipeline_load_error(self, mock_load_pickle):
        """Test prediction when pipeline loading fails."""
        mock_load_pickle.side_effect = FileNotFoundError("Pipeline file not found")

        test_data = pd.DataFrame(
            {
                "Sex": ["M"],
                "Length": [0.5],
                "Diameter": [0.4],
                "Height": [0.1],
                "Whole weight": [0.2],
                "Shucked weight": [0.1],
                "Viscera weight": [0.05],
                "Shell weight": [0.07],
            }
        )

        with pytest.raises(FileNotFoundError):
            predict(test_data)

    @patch("src.modelling.predicting.load_pickle")
    def test_predict_pipeline_prediction_error(self, mock_load_pickle):
        """Test prediction when pipeline prediction fails."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict.side_effect = ValueError("Invalid input data")
        mock_load_pickle.return_value = mock_pipeline

        test_data = pd.DataFrame(
            {
                "Sex": ["M"],
                "Length": [0.5],
                "Diameter": [0.4],
                "Height": [0.1],
                "Whole weight": [0.2],
                "Shucked weight": [0.1],
                "Viscera weight": [0.05],
                "Shell weight": [0.07],
            }
        )

        with pytest.raises(ValueError):
            predict(test_data)
