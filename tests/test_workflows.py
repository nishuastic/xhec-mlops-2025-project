from src.modelling.workflows import train_workflow


def test_train_workflow_runs():
    """Run the full training workflow."""
    result = train_workflow()
    assert "metrics" in result
    assert "model" in result
    assert isinstance(result["metrics"]["RMSE"], float)
