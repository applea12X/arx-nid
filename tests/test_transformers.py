"""
Unit tests for feature transformers.

Tests ensure that our feature engineering pipeline produces
correct outputs and handles edge cases properly.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from arx_nid.features.transformers import (
    RollingStats,
    FlowFeatures,
    CategoricalEncoder,
    FlowPreprocessor,
)


@pytest.fixture
def sample_flow_data():
    """Create sample flow data for testing."""
    # Create a realistic test dataset
    base_time = datetime(2025, 1, 1)

    data = {
        "ts": [base_time + timedelta(seconds=i * 2) for i in range(10)],
        "uid": [f"C{i:03d}" for i in range(10)],
        "id.orig_h": ["192.168.1.10"] * 5 + ["192.168.1.20"] * 5,
        "id.resp_h": ["10.0.0.50"] * 10,
        "id.orig_p": [12345] * 10,
        "id.resp_p": [80] * 10,
        "proto": ["tcp"] * 10,
        "service": ["http"] * 8 + ["https"] * 2,
        "orig_bytes": [100, 200, 300, 400, 500, 150, 250, 350, 450, 550],
        "resp_bytes": [50, 100, 150, 200, 250, 75, 125, 175, 225, 275],
        "orig_pkts": [10, 20, 30, 40, 50, 15, 25, 35, 45, 55],
        "resp_pkts": [5, 10, 15, 20, 25, 8, 13, 18, 23, 28],
        "duration": [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5],
        "conn_state": ["SF"] * 8 + ["S0"] * 2,
    }

    return pd.DataFrame(data)


class TestRollingStats:
    """Test rolling statistics transformer."""

    def test_basic_functionality(self, sample_flow_data):
        """Test basic rolling stats computation."""
        transformer = RollingStats(span_s=5, stats=["mean"])
        result = transformer.fit_transform(sample_flow_data)

        # Check that new columns are added
        assert "orig_bytes_mean_5s" in result.columns
        assert "resp_bytes_mean_5s" in result.columns

        # Check that original data is preserved
        assert len(result) == len(sample_flow_data)
        assert all(col in result.columns for col in sample_flow_data.columns)

    def test_multiple_stats(self, sample_flow_data):
        """Test multiple statistics computation."""
        transformer = RollingStats(span_s=5, stats=["mean", "std", "count"])
        result = transformer.fit_transform(sample_flow_data)

        # Check all requested stats are computed
        expected_cols = [
            "orig_bytes_mean_5s",
            "orig_bytes_std_5s",
            "orig_bytes_count_5s",
            "resp_bytes_mean_5s",
            "resp_bytes_std_5s",
            "resp_bytes_count_5s",
        ]

        for col in expected_cols:
            assert col in result.columns

    def test_different_window_sizes(self, sample_flow_data):
        """Test different window sizes."""
        for window_s in [3, 5, 10]:
            transformer = RollingStats(span_s=window_s)
            result = transformer.fit_transform(sample_flow_data)

            expected_col = f"orig_bytes_mean_{window_s}s"
            assert expected_col in result.columns

    def test_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        transformer = RollingStats(span_s=5)
        empty_df = pd.DataFrame(
            columns=["ts", "id.orig_h", "id.resp_h", "orig_bytes", "resp_bytes"]
        )

        result = transformer.fit_transform(empty_df)
        assert len(result) == 0
        assert all(col in result.columns for col in empty_df.columns)


class TestFlowFeatures:
    """Test flow feature engineering."""

    def test_derived_features(self, sample_flow_data):
        """Test that all derived features are created correctly."""
        transformer = FlowFeatures()
        result = transformer.fit_transform(sample_flow_data)

        # Check new feature columns
        expected_features = [
            "total_bytes",
            "flow_ratio",
            "total_pkts",
            "pkt_size_avg",
            "orig_rate",
            "resp_rate",
            "byte_asymmetry",
            "pkt_asymmetry",
        ]

        for feature in expected_features:
            assert feature in result.columns

    def test_ratio_calculations(self, sample_flow_data):
        """Test specific ratio calculations."""
        transformer = FlowFeatures()
        result = transformer.fit_transform(sample_flow_data)

        # Test flow ratio calculation (should handle division by zero)
        assert all(np.isfinite(result["flow_ratio"]))

        # Test total bytes calculation
        expected_total = sample_flow_data["orig_bytes"] + sample_flow_data["resp_bytes"]
        pd.testing.assert_series_equal(
            result["total_bytes"], expected_total, check_names=False
        )

    def test_zero_duration_handling(self):
        """Test handling of zero duration flows."""
        data = {
            "orig_bytes": [100, 200],
            "resp_bytes": [50, 100],
            "orig_pkts": [10, 20],
            "resp_pkts": [5, 10],
            "duration": [0.0, 1.0],  # Include zero duration
        }
        df = pd.DataFrame(data)

        transformer = FlowFeatures()
        result = transformer.fit_transform(df)

        # Rates should be finite (no division by zero)
        assert all(np.isfinite(result["orig_rate"]))
        assert all(np.isfinite(result["resp_rate"]))


class TestCategoricalEncoder:
    """Test categorical encoding."""

    def test_basic_encoding(self, sample_flow_data):
        """Test basic categorical encoding functionality."""
        categorical_cols = ["proto", "service", "conn_state"]
        encoder = CategoricalEncoder(categorical_cols=categorical_cols)

        encoder.fit(sample_flow_data)
        result = encoder.transform(sample_flow_data)

        # Original categorical columns should be removed
        for col in categorical_cols:
            assert col not in result.columns

        # One-hot encoded columns should be present
        # The encoder uses different naming convention
        encoded_cols = [
            col
            for col in result.columns
            if col.startswith(("proto_", "service_", "conn_state_"))
        ]
        assert len(encoded_cols) > 0

    def test_unknown_categories(self, sample_flow_data):
        """Test handling of unknown categories."""
        encoder = CategoricalEncoder(categorical_cols=["proto"])
        encoder.fit(sample_flow_data)

        # Create test data with unknown category
        test_data = sample_flow_data.copy()
        test_data.loc[0, "proto"] = "unknown_protocol"

        # Should not raise an error with handle_unknown='ignore'
        result = encoder.transform(test_data)
        assert len(result) == len(test_data)

    def test_missing_values(self):
        """Test handling of missing categorical values."""
        data = {
            "proto": ["tcp", "udp", None, "tcp"],
            "service": ["http", None, "https", "http"],
        }
        df = pd.DataFrame(data)

        encoder = CategoricalEncoder(categorical_cols=["proto", "service"])
        encoder.fit(df)
        result = encoder.transform(df)

        # Should handle missing values gracefully
        assert len(result) == len(df)


class TestFlowPreprocessor:
    """Test the complete preprocessing pipeline."""

    def test_full_pipeline(self, sample_flow_data):
        """Test the complete preprocessing pipeline."""
        preprocessor = FlowPreprocessor(
            rolling_window_s=5,
            rolling_stats=["mean"],
            categorical_cols=["proto", "service"],
            numeric_cols=["orig_bytes", "resp_bytes", "duration"],
        )

        result = preprocessor.fit_transform(sample_flow_data)

        # Check that processing completed successfully
        assert len(result) == len(sample_flow_data)
        assert len(result.columns) > len(sample_flow_data.columns)  # New features added

    def test_pipeline_consistency(self, sample_flow_data):
        """Test that fit + transform equals fit_transform."""
        preprocessor = FlowPreprocessor()

        # Method 1: fit_transform
        result1 = preprocessor.fit_transform(sample_flow_data.copy())

        # Method 2: fit then transform
        preprocessor2 = FlowPreprocessor()
        preprocessor2.fit(sample_flow_data.copy())
        result2 = preprocessor2.transform(sample_flow_data.copy())

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_pipeline_reusability(self, sample_flow_data):
        """Test that fitted pipeline can be reused on new data."""
        preprocessor = FlowPreprocessor()
        preprocessor.fit(sample_flow_data)

        # Create slightly different test data
        test_data = sample_flow_data.copy()
        test_data["orig_bytes"] = test_data["orig_bytes"] * 2

        # Should be able to transform new data
        result = preprocessor.transform(test_data)
        assert len(result) == len(test_data)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_minimal_dataframe(self):
        """Test with minimal required columns."""
        data = {
            "ts": [datetime(2025, 1, 1)],
            "id.orig_h": ["1.1.1.1"],
            "id.resp_h": ["2.2.2.2"],
            "orig_bytes": [100],
            "resp_bytes": [50],
        }
        df = pd.DataFrame(data)

        transformer = RollingStats(span_s=5)
        result = transformer.fit_transform(df)

        # Should handle single row gracefully
        assert len(result) == 1

    def test_large_values(self):
        """Test handling of large numeric values."""
        data = {
            "ts": [datetime(2025, 1, 1), datetime(2025, 1, 1, 0, 0, 10)],
            "id.orig_h": ["1.1.1.1", "1.1.1.1"],
            "id.resp_h": ["2.2.2.2", "2.2.2.2"],
            "orig_bytes": [1e12, 2e12],  # Very large values
            "resp_bytes": [5e11, 1e12],
            "orig_pkts": [1000000, 2000000],
            "resp_pkts": [500000, 1000000],
            "duration": [1000.0, 2000.0],
        }
        df = pd.DataFrame(data)

        flow_features = FlowFeatures()
        result = flow_features.fit_transform(df)

        # Should handle large values without overflow
        assert all(np.isfinite(result["total_bytes"]))
        assert all(np.isfinite(result["flow_ratio"]))


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
