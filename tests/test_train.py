"""Tests für src/train.py — Multi-Label-Trainings- und Evaluationslogik.

compute_pos_weight() und _metrics_per_label()/find_optimal_thresholds_multilabel()
sind hier am kritischsten, da sie das Multi-Label-Verhalten festlegen
(Vektor/DataFrame statt Skalar je Label).
"""
import math

import numpy as np
import pytest
import torch

from src.data import TARGET_COLUMNS, MultiLabelDataset, encode_type
from src.train import (
    _metrics_per_label,
    compute_pos_weight,
    find_optimal_threshold,
)


def test_compute_pos_weight_returns_vector_not_scalar(synthetic_df):
    """Ein Gewicht je Label, nicht ein einzelner Skalar für das gesamte Modell."""
    encoded = encode_type(synthetic_df)
    dataset = MultiLabelDataset(encoded)
    pos_weight = compute_pos_weight(dataset)

    assert pos_weight.shape == (len(TARGET_COLUMNS),)
    # Jedes Gewicht muss positiv sein (n_negative/n_positive bei seltenen
    # positiven Fällen immer > 1)
    assert torch.all(pos_weight > 0)


def test_compute_pos_weight_handles_label_with_zero_positives():
    """Regressionstest gegen Division durch 0: falls ein Label im
    Trainingssplit zufällig keine einzige positive Beobachtung enthält
    (realistisch bei TWF mit nur wenigen Fällen im Gesamtdatensatz)."""
    y = torch.zeros(50, 4)
    y[:, 0] = 1  # Label 0 hat positive Fälle, Labels 1-3 haben keine

    class FakeDataset:
        pass

    ds = FakeDataset()
    ds.y = y
    pos_weight = compute_pos_weight(ds)

    assert torch.isfinite(pos_weight).all()
    assert not torch.isnan(pos_weight).any()


def test_metrics_per_label_returns_one_row_per_target():
    rng = np.random.default_rng(0)
    n = 100
    y_true = rng.integers(0, 2, size=(n, len(TARGET_COLUMNS)))
    y_prob = rng.uniform(0, 1, size=(n, len(TARGET_COLUMNS)))

    metrics, _ = _metrics_per_label(y_true, y_prob, threshold=0.5, target_columns=TARGET_COLUMNS)

    assert list(metrics.index) == TARGET_COLUMNS
    assert set(metrics.columns) >= {"precision", "recall", "f1", "roc_auc", "pr_auc", "n_positive"}






def test_confusion_matrix_cells_match_metrics_columns():
    """tn/fp/fn/tp in der metrics-Tabelle müssen exakt mit den Zellen der
    zugehörigen 2x2-Konfusionsmatrix übereinstimmen -- beide werden aus
    denselben Vorhersagen abgeleitet, dürfen also nie auseinanderlaufen."""
    y_true = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
    y_prob = np.array([[0.2, 0.8], [0.9, 0.3], [0.1, 0.4], [0.6, 0.1]])
    target_columns = ["A", "B"]

    metrics, matrices = _metrics_per_label(y_true, y_prob, threshold=0.5, target_columns=target_columns)

    for label in target_columns:
        tn, fp, fn, tp = matrices[label].ravel()
        row = metrics.loc[label]
        assert (row["tn"], row["fp"], row["fn"], row["tp"]) == (tn, fp, fn, tp)
        # Konsistenzcheck: Summe aller vier Zellen muss der Beobachtungszahl entsprechen
        assert tn + fp + fn + tp == len(y_true)




def test_find_optimal_threshold_single_label_perfect_separation():
    """Bestaetigt, dass find_optimal_threshold() bei perfekt trennbaren
    Daten F1=1,0 liefert."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold, f1 = find_optimal_threshold(y_true, y_prob)
    assert math.isclose(f1, 1.0, rel_tol=1e-6)
    assert 0.3 < threshold <= 0.7


def test_metrics_per_label_handles_labels_without_positives_or_predictions():
    """Division durch 0: Ein Label ohne positive Fälle und eines, für das nie
    ein Ausfall vorhergesagt wird, dürfen weder Fehler noch inf/NaN in
    Precision, Recall und F1 erzeugen (AUC ist ohne Positive nicht definiert)."""
    y_true = np.array([[0, 1], [0, 1], [0, 0], [0, 0]])
    y_prob = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.1], [0.1, 0.2]])  # nie >= 0,5

    metrics, _ = _metrics_per_label(y_true, y_prob, threshold=0.5, target_columns=["A", "B"])

    assert np.isfinite(metrics[["precision", "recall", "f1"]].to_numpy()).all()
    assert (metrics["f1"] == 0).all()
    assert np.isnan(metrics.loc["A", "roc_auc"])


def test_metrics_per_label_applies_each_threshold_to_its_own_label():
    """Jedes Label wird mit seinem eigenen Schwellenwert bewertet (nicht mit
    einem gemeinsamen, nicht vertauscht): Wahrscheinlichkeit 0,6 ist bei
    Schwellenwert 0,5 positiv und bei 0,7 negativ."""
    y_true = np.array([[1, 1], [0, 0]])
    y_prob = np.array([[0.6, 0.6], [0.1, 0.1]])

    metrics, _ = _metrics_per_label(y_true, y_prob, threshold=[0.5, 0.7], target_columns=["A", "B"])

    assert list(metrics["threshold"]) == [0.5, 0.7]
    assert (metrics.loc["A", "tp"], metrics.loc["A", "fn"]) == (1, 0)
    assert (metrics.loc["B", "tp"], metrics.loc["B", "fn"]) == (0, 1)

