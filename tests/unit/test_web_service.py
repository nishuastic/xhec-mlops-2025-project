"""Unit tests for web service module."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.web_service.lib.models import AbaloneInput, PredObj
from src.web_service.main import app
from src.web_service.utils import load_pickle

pytestmark = pytest.mark.unit


class TestWebServiceModels:
    """Test cases for web service models."""

    def test_abalone_input_valid_data(self):
        """Test AbaloneInput with valid data."""
        data = {
            "Sex": "M",
            "Length": 0.5,
            "Diameter": 0.4,
            "Height": 0.1,
            "Whole weight": 0.2,
            "Shucked weight": 0.1,
            "Viscera weight": 0.05,
            "Shell weight": 0.07,
        }

        abalone = AbaloneInput(**data)
        assert abalone.Sex == "M"
        assert abalone.Length == 0.5
        assert abalone.Diameter == 0.4
        assert abalone.Height == 0.1
        assert abalone.Whole_weight == 0.2
        assert abalone.Shucked_weight == 0.1
        assert abalone.Viscera_weight == 0.05
        assert abalone.Shell_weight == 0.07

    def test_abalone_input_invalid_sex(self):
        """Test AbaloneInput with invalid sex."""
        data = {
            "Sex": "X",  # Invalid sex
            "Length": 0.5,
            "Diameter": 0.4,
            "Height": 0.1,
            "Whole weight": 0.2,
            "Shucked weight": 0.1,
            "Viscera weight": 0.05,
            "Shell weight": 0.07,
        }

        with pytest.raises(ValueError):
            AbaloneInput(**data)

    def test_abalone_input_missing_fields(self):
        """Test AbaloneInput with missing required fields."""
        data = {
            "Sex": "M",
            "Length": 0.5,
            # Missing other required fields
        }

        with pytest.raises(ValueError):
            AbaloneInput(**data)

    def test_pred_obj_valid(self):
        """Test PredObj with valid data."""
        pred = PredObj(Age=5.2)
        assert pred.Age == 5.2

    def test_pred_obj_negative_age(self):
        """Test PredObj with negative age (should still work)."""
        pred = PredObj(Age=-1.0)
        assert pred.Age == -1.0


class TestWebServiceUtils:
    """Test cases for web service utils."""

    @patch("src.web_service.utils.logger")
    def test_load_pickle_success(self, mock_logger, tmp_path):
        """Test successful pickle loading."""
        import pickle

        test_data = {"test": "data", "number": 42}
        filepath = tmp_path / "test.pkl"

        with open(filepath, "wb") as f:
            pickle.dump(test_data, f)

        result = load_pickle(str(filepath))

        assert result == test_data
        mock_logger.info.assert_called_once_with(f"Loaded: {filepath}")

    def test_load_pickle_file_not_found(self, tmp_path):
        """Test load_pickle with non-existent file."""
        filepath = tmp_path / "nonexistent.pkl"

        with pytest.raises(FileNotFoundError):
            load_pickle(str(filepath))

    @patch("src.web_service.utils.logger")
    def test_load_pickle_corrupted_file(self, mock_logger, tmp_path):
        """Test load_pickle with corrupted file."""
        filepath = tmp_path / "corrupted.pkl"
        filepath.write_text("corrupted data")

        with pytest.raises(Exception):  # Should raise some kind of pickle error
            load_pickle(str(filepath))


class TestWebServiceAPI:
    """Test cases for web service API endpoints."""

    def test_home_endpoint(self):
        """Test home endpoint."""
        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        assert response.json() == {"health_check": "App up and running!"}

    @patch("src.web_service.main.load_pickle")
    def test_predict_endpoint_success(self, mock_load_pickle):
        """Test predict endpoint with successful prediction."""
        # Setup mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = [5.2]
        mock_load_pickle.return_value = mock_pipeline

        client = TestClient(app)

        payload = {
            "Sex": "M",
            "Length": 0.5,
            "Diameter": 0.4,
            "Height": 0.1,
            "Whole weight": 0.2,
            "Shucked weight": 0.1,
            "Viscera weight": 0.05,
            "Shell weight": 0.07,
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 201
        assert response.json() == {"Age": 5.2}
        mock_load_pickle.assert_called_once()

    @patch("src.web_service.main.load_pickle")
    def test_predict_endpoint_invalid_input(self, mock_load_pickle):
        """Test predict endpoint with invalid input."""
        client = TestClient(app)

        # Invalid payload - missing required fields
        payload = {
            "Sex": "M",
            "Length": 0.5,
            # Missing other required fields
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 422  # Validation error
        mock_load_pickle.assert_not_called()

    @patch("src.web_service.main.load_pickle")
    def test_predict_endpoint_pipeline_error(self, mock_load_pickle):
        """Test predict endpoint when pipeline loading fails."""
        mock_load_pickle.side_effect = FileNotFoundError("Pipeline not found")

        client = TestClient(app)

        payload = {
            "Sex": "M",
            "Length": 0.5,
            "Diameter": 0.4,
            "Height": 0.1,
            "Whole weight": 0.2,
            "Shucked weight": 0.1,
            "Viscera weight": 0.05,
            "Shell weight": 0.07,
        }

        # The endpoint doesn't have error handling, so it will raise the exception
        with pytest.raises(FileNotFoundError):
            client.post("/predict", json=payload)

    @patch("src.web_service.main.load_pickle")
    def test_predict_endpoint_prediction_error(self, mock_load_pickle):
        """Test predict endpoint when prediction fails."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict.side_effect = ValueError("Invalid input data")
        mock_load_pickle.return_value = mock_pipeline

        client = TestClient(app)

        payload = {
            "Sex": "M",
            "Length": 0.5,
            "Diameter": 0.4,
            "Height": 0.1,
            "Whole weight": 0.2,
            "Shucked weight": 0.1,
            "Viscera weight": 0.05,
            "Shell weight": 0.07,
        }

        # The endpoint doesn't have error handling, so it will raise the exception
        with pytest.raises(ValueError):
            client.post("/predict", json=payload)

    def test_predict_endpoint_with_different_sex_values(self):
        """Test predict endpoint with different sex values."""
        with patch("src.web_service.main.load_pickle") as mock_load_pickle:
            mock_pipeline = MagicMock()
            mock_pipeline.predict.return_value = [5.2]
            mock_load_pickle.return_value = mock_pipeline

            client = TestClient(app)

            # Test with Female
            payload_f = {
                "Sex": "F",
                "Length": 0.6,
                "Diameter": 0.45,
                "Height": 0.15,
                "Whole weight": 0.25,
                "Shucked weight": 0.12,
                "Viscera weight": 0.06,
                "Shell weight": 0.08,
            }

            response = client.post("/predict", json=payload_f)
            assert response.status_code == 201

            # Test with Infant
            payload_i = {
                "Sex": "I",
                "Length": 0.4,
                "Diameter": 0.35,
                "Height": 0.08,
                "Whole weight": 0.15,
                "Shucked weight": 0.09,
                "Viscera weight": 0.04,
                "Shell weight": 0.05,
            }

            response = client.post("/predict", json=payload_i)
            assert response.status_code == 201
