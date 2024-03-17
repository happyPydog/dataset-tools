"""Tests for pandas dataset."""

from dataset_tools.pandas_dataset import NumericalDataset


def test_numerical_dataset():
    dataset = NumericalDataset(sample_size=100, feature_size=10, missing_ratio=0.5)
    df = dataset.generate()
    assert len(df) == 100
    assert len(df.columns) == 12
    assert "id" in df.columns
    assert "label" in df.columns
