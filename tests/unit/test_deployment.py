"""Unit tests for deployment module."""

import pytest

pytestmark = pytest.mark.unit


class TestDeployment:
    """Test cases for deployment module."""

    def test_train_weekly_import(self):
        """Test that train_weekly can be imported."""
        from src.modelling.deployment import train_weekly

        assert train_weekly is not None

    def test_train_weekly_is_deployment(self):
        """Test that train_weekly is a deployment object."""
        from src.modelling.deployment import train_weekly

        # Verify it's a deployment object
        assert hasattr(train_weekly, "name")
        assert hasattr(train_weekly, "version")
        assert hasattr(train_weekly, "tags")
        assert hasattr(train_weekly, "parameters")

    def test_train_weekly_properties(self):
        """Test that train_weekly has expected properties."""
        from src.modelling.deployment import train_weekly

        # Check deployment properties
        assert train_weekly.name == "abalone-retrain-weekly"
        assert train_weekly.version == "0.1.0"
        assert "training" in train_weekly.tags
        assert "weekly" in train_weekly.tags
        assert "retrain" in train_weekly.tags

    def test_train_weekly_parameters(self):
        """Test that train_weekly has correct parameters."""
        from src.modelling.deployment import train_weekly

        expected_params = {
            "input_filepath": "data/abalone.csv",
            "artifacts_dirpath": "models",
            "model_type": "xgboost",
        }

        assert train_weekly.parameters == expected_params

    def test_train_weekly_tags(self):
        """Test that train_weekly has correct tags."""
        from src.modelling.deployment import train_weekly

        expected_tags = ["training", "weekly", "retrain"]

        assert train_weekly.tags == expected_tags
        assert len(train_weekly.tags) == 3

    def test_deployment_imports(self):
        """Test that all required imports are available."""
        from prefect import serve

        from src.modelling.deployment import train_weekly

        # Verify imports work correctly
        assert train_weekly is not None
        assert serve is not None
        assert callable(serve)

    def test_deployment_module_structure(self):
        """Test the overall structure of the deployment module."""
        import src.modelling.deployment

        # Verify the module has the expected attributes
        assert hasattr(src.modelling.deployment, "train_weekly")
        assert hasattr(src.modelling.deployment, "serve")

        # Verify train_weekly is a deployment object
        from src.modelling.deployment import train_weekly

        assert hasattr(train_weekly, "name")
        assert hasattr(train_weekly, "version")
        assert hasattr(train_weekly, "tags")
        assert hasattr(train_weekly, "parameters")
