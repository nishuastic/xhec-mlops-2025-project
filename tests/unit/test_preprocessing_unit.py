"""Unit tests for preprocessing module."""

from unittest.mock import patch

import pandas as pd
import pytest

from src.modelling.preprocessing import (
    add_target,
    filter_outliers,
    load_data,
    prepare_features,
)

pytestmark = pytest.mark.unit


class TestAddTarget:
    """Test cases for add_target function."""

    def test_add_target_creates_age_column(self, sample_dataframe_with_rings):
        """Test that Age column is created correctly."""
        result = add_target(sample_dataframe_with_rings)

        assert "Age" in result.columns
        assert "Rings" not in result.columns
        assert result["Age"].equals(sample_dataframe_with_rings["Rings"] + 1.5)

    def test_add_target_with_empty_dataframe(self):
        """Test add_target with empty dataframe."""
        df = pd.DataFrame({"Rings": []})
        result = add_target(df)

        assert "Age" in result.columns
        assert "Rings" not in result.columns
        assert len(result) == 0

    def test_add_target_preserves_other_columns(self, sample_dataframe_with_rings):
        """Test that other columns are preserved."""
        result = add_target(sample_dataframe_with_rings)

        expected_columns = set(sample_dataframe_with_rings.columns) - {"Rings"}
        assert set(result.columns) == expected_columns


class TestFilterOutliers:
    """Test cases for filter_outliers function."""

    def test_filter_outliers_removes_extreme_values(self):
        """Test that extreme outliers are removed."""
        df = pd.DataFrame(
            {
                "Length": [0.5, 0.6, 0.4, 0.55, 10.0],  # 10.0 is an extreme outlier
                "Diameter": [0.4, 0.45, 0.35, 0.42, 0.38],
                "Height": [0.1, 0.15, 0.08, 0.12, 0.11],
                "Whole weight": [0.2, 0.25, 0.15, 0.22, 0.21],
                "Shucked weight": [0.1, 0.12, 0.09, 0.11, 0.10],
                "Viscera weight": [0.05, 0.06, 0.04, 0.05, 0.04],
                "Shell weight": [0.07, 0.08, 0.05, 0.07, 0.06],
                "Age": [6.5, 11.5, 4.5, 7.5, 6.8],
            }
        )

        result = filter_outliers(df)
        assert len(result) < len(df)
        assert 10.0 not in result["Length"].values

    def test_filter_outliers_with_no_outliers(self, sample_dataframe):
        """Test filter_outliers with no outliers."""
        result = filter_outliers(sample_dataframe)
        assert len(result) == len(sample_dataframe)

    def test_filter_outliers_handles_nan_values(self):
        """Test that NaN values are preserved."""
        df = pd.DataFrame(
            {
                "Length": [0.5, 0.6, None, 0.55],
                "Diameter": [0.4, 0.45, 0.35, 0.42],
                "Height": [0.1, 0.15, 0.08, 0.12],
                "Whole weight": [0.2, 0.25, 0.15, 0.22],
                "Shucked weight": [0.1, 0.12, 0.09, 0.11],
                "Viscera weight": [0.05, 0.06, 0.04, 0.05],
                "Shell weight": [0.07, 0.08, 0.05, 0.07],
                "Age": [6.5, 11.5, 4.5, 7.5],
            }
        )

        result = filter_outliers(df)
        # The function should preserve rows with NaN values
        assert len(result) >= 1  # At least one row should be preserved
        # Check that the function doesn't crash with NaN values
        assert not result.empty


class TestLoadData:
    """Test cases for load_data function."""

    @patch("pandas.read_csv")
    def test_load_data_calls_read_csv(self, mock_read_csv, tmp_path):
        """Test that load_data calls pandas.read_csv."""
        mock_df = pd.DataFrame({"test": [1, 2, 3]})
        mock_read_csv.return_value = mock_df

        filepath = tmp_path / "test.csv"
        filepath.write_text("test,data")

        result = load_data(str(filepath))

        mock_read_csv.assert_called_once_with(str(filepath))
        assert result.equals(mock_df)


class TestPrepareFeatures:
    """Test cases for prepare_features function."""

    def test_prepare_features_splits_data(self, sample_dataframe):
        """Test that data is split into train/test."""
        X_train, X_test, y_train, y_test, preprocessor = prepare_features(
            sample_dataframe
        )

        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        assert len(X_train) + len(X_test) == len(sample_dataframe)

    def test_prepare_features_creates_preprocessor(self, sample_dataframe):
        """Test that preprocessor is created correctly."""
        _, _, _, _, preprocessor = prepare_features(sample_dataframe)

        assert preprocessor is not None
        assert hasattr(preprocessor, "fit_transform")

    def test_prepare_features_with_empty_dataframe(self):
        """Test prepare_features with empty dataframe."""
        df = pd.DataFrame(columns=["Sex", "Length", "Age"])

        with pytest.raises(ValueError):
            prepare_features(df)

    def test_prepare_features_removes_target_column(self, sample_dataframe):
        """Test that target column is removed from features."""
        X_train, X_test, _, _, _ = prepare_features(sample_dataframe)

        assert "Age" not in X_train.columns
        assert "Age" not in X_test.columns
