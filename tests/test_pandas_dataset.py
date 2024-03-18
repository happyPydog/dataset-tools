"""Tests for pandas dataset."""

import pandas as pd

from dataset_tools.pandas_dataset import NumericalDataset, TimeSeriesDataset


def test_numerical_dataset():
    dataset = NumericalDataset(sample_size=100, feature_size=10, missing_ratio=0.5)
    df = dataset.generate()
    assert len(df) == 100
    assert len(df.columns) == 12
    assert "id" in df.columns
    assert "label" in df.columns


def test_timeseries_dataset():
    dataset = TimeSeriesDataset(
        sample_size=100,
        feature_size=10,
        start_date="2020-01-01",
        end_date="2020-01-10",
        missing_ratio=0.5,
    )
    df = dataset.generate()
    assert len(df) == 100 * 10
    assert len(df.columns) == 13
    assert "id" in df.columns
    assert "label" in df.columns
