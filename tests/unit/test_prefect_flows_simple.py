"""Simple unit tests for prefect_flows module."""

import pytest

pytestmark = pytest.mark.unit


class TestPrefectFlowsImports:
    """Test that prefect_flows module can be imported and has expected functions."""

    def test_prefect_flows_imports(self):
        """Test that all required functions can be imported."""
        from src.modelling.prefect_flows import (
            t_add_target,
            t_evaluate_model,
            t_filter_outliers,
            t_load_data,
            t_prepare_features,
            t_save_artifacts,
            t_train_model,
            train_flow,
        )

        # Verify all functions are callable
        assert callable(t_load_data)
        assert callable(t_add_target)
        assert callable(t_filter_outliers)
        assert callable(t_prepare_features)
        assert callable(t_train_model)
        assert callable(t_evaluate_model)
        assert callable(t_save_artifacts)
        assert callable(train_flow)

    def test_train_flow_is_flow(self):
        """Test that train_flow is a Prefect flow."""
        from src.modelling.prefect_flows import train_flow

        # Verify train_flow is a flow object
        assert hasattr(train_flow, "name")
        assert train_flow.name == "abalone-train-flow"

    def test_prefect_flows_module_structure(self):
        """Test the overall structure of the prefect_flows module."""
        import src.modelling.prefect_flows

        # Verify the module has the expected attributes
        assert hasattr(src.modelling.prefect_flows, "t_load_data")
        assert hasattr(src.modelling.prefect_flows, "t_add_target")
        assert hasattr(src.modelling.prefect_flows, "t_filter_outliers")
        assert hasattr(src.modelling.prefect_flows, "t_prepare_features")
        assert hasattr(src.modelling.prefect_flows, "t_train_model")
        assert hasattr(src.modelling.prefect_flows, "t_evaluate_model")
        assert hasattr(src.modelling.prefect_flows, "t_save_artifacts")
        assert hasattr(src.modelling.prefect_flows, "train_flow")

    def test_prefect_flows_main_execution(self):
        """Test prefect_flows main execution."""
        # Test that the module can be imported and the function exists
        from src.modelling.prefect_flows import train_flow

        assert train_flow is not None
        assert callable(train_flow)

    def test_prefect_flows_config_imports(self):
        """Test that config imports work correctly."""
        from src.modelling.prefect_flows import DATA_DIR, MODEL_TYPE

        # Verify imports work correctly
        assert DATA_DIR is not None
        assert MODEL_TYPE is not None

    def test_prefect_flows_function_imports(self):
        """Test that function imports work correctly."""
        from src.modelling.prefect_flows import (
            add_target,
            evaluate_model,
            filter_outliers,
            load_data,
            prepare_features,
            save_artifacts,
            train_model,
        )

        # Verify all functions are callable
        assert callable(load_data)
        assert callable(add_target)
        assert callable(filter_outliers)
        assert callable(prepare_features)
        assert callable(train_model)
        assert callable(evaluate_model)
        assert callable(save_artifacts)

    def test_prefect_flows_prefect_imports(self):
        """Test that Prefect imports work correctly."""
        from src.modelling.prefect_flows import flow, get_run_logger, task

        # Verify imports work correctly
        assert flow is not None
        assert task is not None
        assert callable(get_run_logger)
