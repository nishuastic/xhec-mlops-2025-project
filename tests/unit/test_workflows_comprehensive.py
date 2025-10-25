"""Comprehensive unit tests for workflows module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


class TestWorkflows:
    """Test cases for workflows module."""

    @patch("src.modelling.workflows.save_artifacts")
    @patch("src.modelling.workflows.evaluate_model")
    @patch("src.modelling.workflows.train_model")
    @patch("src.modelling.workflows.prepare_features")
    @patch("src.modelling.workflows.filter_outliers")
    @patch("src.modelling.workflows.add_target")
    @patch("src.modelling.workflows.load_data")
    @patch("src.modelling.workflows.logger")
    def test_train_workflow_with_defaults(
        self,
        mock_logger,
        mock_load_data,
        mock_add_target,
        mock_filter_outliers,
        mock_prepare_features,
        mock_train_model,
        mock_evaluate_model,
        mock_save_artifacts,
    ):
        """Test train_workflow with default parameters."""
        # Setup mocks
        mock_df = pd.DataFrame({"test": [1, 2, 3]})
        mock_load_data.return_value = mock_df
        mock_add_target.return_value = mock_df
        mock_filter_outliers.return_value = mock_df

        X_train, X_test, y_train, y_test, preprocessor = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.Series(),
            pd.Series(),
            MagicMock(),
        )
        mock_prepare_features.return_value = (
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
        )

        mock_pipe = MagicMock()
        mock_train_model.return_value = mock_pipe

        mock_metrics = {"MAE": 0.5, "RMSE": 0.7, "R2": 0.8}
        mock_evaluate_model.return_value = mock_metrics

        # Call the workflow
        from src.modelling.workflows import train_workflow

        result = train_workflow()

        # Assertions
        assert result == {"model": mock_pipe, "metrics": mock_metrics}

        # Verify logger calls
        mock_logger.info.assert_called()
        mock_logger.success.assert_called()

        # Verify function calls
        mock_load_data.assert_called_once()
        mock_add_target.assert_called_once()
        mock_filter_outliers.assert_called_once()
        mock_prepare_features.assert_called_once()
        mock_train_model.assert_called_once()
        mock_evaluate_model.assert_called_once()
        mock_save_artifacts.assert_called_once()

    @patch("src.modelling.workflows.save_artifacts")
    @patch("src.modelling.workflows.evaluate_model")
    @patch("src.modelling.workflows.train_model")
    @patch("src.modelling.workflows.prepare_features")
    @patch("src.modelling.workflows.filter_outliers")
    @patch("src.modelling.workflows.add_target")
    @patch("src.modelling.workflows.load_data")
    @patch("src.modelling.workflows.logger")
    def test_train_workflow_with_custom_parameters(
        self,
        mock_logger,
        mock_load_data,
        mock_add_target,
        mock_filter_outliers,
        mock_prepare_features,
        mock_train_model,
        mock_evaluate_model,
        mock_save_artifacts,
    ):
        """Test train_workflow with custom parameters."""
        # Setup mocks
        mock_df = pd.DataFrame({"test": [1, 2, 3]})
        mock_load_data.return_value = mock_df
        mock_add_target.return_value = mock_df
        mock_filter_outliers.return_value = mock_df

        X_train, X_test, y_train, y_test, preprocessor = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.Series(),
            pd.Series(),
            MagicMock(),
        )
        mock_prepare_features.return_value = (
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
        )

        mock_pipe = MagicMock()
        mock_train_model.return_value = mock_pipe

        mock_metrics = {"MAE": 0.5, "RMSE": 0.7, "R2": 0.8}
        mock_evaluate_model.return_value = mock_metrics

        # Call the workflow with custom parameters
        from src.modelling.workflows import train_workflow

        result = train_workflow(
            input_filepath="custom_data.csv",
            artifacts_dirpath="custom_models",
            model_type="GradientBoosting",
        )

        # Assertions
        assert result == {"model": mock_pipe, "metrics": mock_metrics}

        # Verify custom parameters were used
        mock_load_data.assert_called_once_with("custom_data.csv")

    @patch("src.modelling.workflows.save_artifacts")
    @patch("src.modelling.workflows.evaluate_model")
    @patch("src.modelling.workflows.train_model")
    @patch("src.modelling.workflows.prepare_features")
    @patch("src.modelling.workflows.filter_outliers")
    @patch("src.modelling.workflows.add_target")
    @patch("src.modelling.workflows.load_data")
    @patch("src.modelling.workflows.logger")
    def test_train_workflow_with_none_parameters(
        self,
        mock_logger,
        mock_load_data,
        mock_add_target,
        mock_filter_outliers,
        mock_prepare_features,
        mock_train_model,
        mock_evaluate_model,
        mock_save_artifacts,
    ):
        """Test train_workflow with None parameters (should use defaults)."""
        # Setup mocks
        mock_df = pd.DataFrame({"test": [1, 2, 3]})
        mock_load_data.return_value = mock_df
        mock_add_target.return_value = mock_df
        mock_filter_outliers.return_value = mock_df

        X_train, X_test, y_train, y_test, preprocessor = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.Series(),
            pd.Series(),
            MagicMock(),
        )
        mock_prepare_features.return_value = (
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
        )

        mock_pipe = MagicMock()
        mock_train_model.return_value = mock_pipe

        mock_metrics = {"MAE": 0.5, "RMSE": 0.7, "R2": 0.8}
        mock_evaluate_model.return_value = mock_metrics

        # Call the workflow with None parameters
        from src.modelling.workflows import train_workflow

        result = train_workflow(
            input_filepath=None, artifacts_dirpath=None, model_type=None
        )

        # Assertions
        assert result == {"model": mock_pipe, "metrics": mock_metrics}

        # Verify default parameters were used
        mock_load_data.assert_called_once()
        mock_save_artifacts.assert_called_once()

    def test_train_workflow_import(self):
        """Test that train_workflow can be imported."""
        from src.modelling.workflows import train_workflow

        assert train_workflow is not None
        assert callable(train_workflow)

    def test_workflows_module_structure(self):
        """Test the overall structure of the workflows module."""
        import src.modelling.workflows

        # Verify the module has the expected attributes
        assert hasattr(src.modelling.workflows, "train_workflow")

        # Verify train_workflow is callable
        from src.modelling.workflows import train_workflow

        assert callable(train_workflow)

    def test_workflows_imports(self):
        """Test that all required imports are available."""
        from src.modelling.workflows import (
            add_target,
            evaluate_model,
            filter_outliers,
            load_data,
            prepare_features,
            save_artifacts,
            train_model,
            train_workflow,
        )

        # Verify all functions are callable
        assert callable(train_workflow)
        assert callable(load_data)
        assert callable(add_target)
        assert callable(filter_outliers)
        assert callable(prepare_features)
        assert callable(train_model)
        assert callable(evaluate_model)
        assert callable(save_artifacts)
