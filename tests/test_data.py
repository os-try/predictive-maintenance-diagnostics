"""Tests für src/data.py — Multi-Label-Datenaufbereitung.

Getestet wird die Infrastruktur (Split-Anteile, kein Data Leakage,
korrekte Tensor-Shapes), nicht die Modellgüte.
"""
import numpy as np
import torch

from src.data import (
    BINARY_TARGET_COLUMN,
    ENCODED_TYPE_COLUMNS,
    EXCLUDED_TARGET,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    MultiLabelDataset,
    apply_scaler,
    encode_type,
    fit_scaler,
    split_data,
)


def test_target_columns_excludes_rnf():
    """RNF darf nicht Teil der vier betrachteten Zielspalten sein."""
    assert EXCLUDED_TARGET not in TARGET_COLUMNS
    assert TARGET_COLUMNS == ["TWF", "HDF", "PWF", "OSF"]


def test_split_ratios_and_no_overlap(synthetic_df):
    n = len(synthetic_df)
    train, val, test = split_data(synthetic_df, test_size=0.15, val_size=0.15)

    assert abs(len(test) / n - 0.15) < 0.05
    assert abs(len(val) / n - 0.15) < 0.05

    idx_train, idx_val, idx_test = set(train.index), set(val.index), set(test.index)
    assert idx_train.isdisjoint(idx_val)
    assert idx_train.isdisjoint(idx_test)
    assert idx_val.isdisjoint(idx_test)


def test_split_stratifies_each_label_individually(synthetic_df):
    """Echte Multi-Label-Stratifizierung (MultilabelStratifiedShuffleSplit):
    jede der vier Zielspalten einzeln sollte in allen drei Splits nahe am
    Gesamtanteil liegen, nicht nur das übergeordnete Binärlabel."""
    train, val, test = split_data(synthetic_df)

    for col in TARGET_COLUMNS:
        overall_rate = synthetic_df[col].mean()
        for split in (train, val, test):
            assert abs(split[col].mean() - overall_rate) < 0.15

    # Zusätzlich weiterhin: das übergeordnete Binärlabel bleibt ebenfalls balanciert
    overall_binary_rate = synthetic_df[BINARY_TARGET_COLUMN].mean()
    for split in (train, val, test):
        assert abs(split[BINARY_TARGET_COLUMN].mean() - overall_binary_rate) < 0.15


def test_scaler_fitted_only_on_train_has_zero_mean_on_train(synthetic_df):
    train, val, test = split_data(synthetic_df)
    scaler = fit_scaler(train)
    scaled_train = apply_scaler(train, scaler)
    means = scaled_train[FEATURE_COLUMNS].mean().values
    assert np.allclose(means, 0, atol=1e-6)


def test_multilabel_dataset_produces_correct_shapes(synthetic_df):
    """Regressionstest für die Kernidee des Projekts: y hat die Form
    (N, 4) statt (N, 1)."""
    encoded = encode_type(synthetic_df)
    dataset = MultiLabelDataset(encoded)

    assert len(dataset) == len(encoded)
    X, y = dataset[0]
    assert X.dtype == torch.float32
    assert y.dtype == torch.float32
    assert X.shape == (len(FEATURE_COLUMNS) + len(ENCODED_TYPE_COLUMNS),)
    assert y.shape == (len(TARGET_COLUMNS),)
