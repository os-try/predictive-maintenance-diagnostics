"""Tests für src/models.py — MLPMultiLabel-Struktur.

Getestet werden Output-Shapes und Wertebereiche (Sanity Checks),
NICHT die Vorhersagegüte.
"""
import torch

from src.models import MLPMultiLabel

INPUT_DIM = 10  # 7 numerische Features + 3 One-Hot-Type-Spalten
N_LABELS = 4  # TWF, HDF, PWF, OSF


def test_output_shape_matches_n_labels():
    """Die Ausgabeschicht hat n_labels statt 1 Neuronen."""
    model = MLPMultiLabel(input_dim=INPUT_DIM, n_labels=N_LABELS)
    x = torch.randn(16, INPUT_DIM)
    out = model(x)
    assert out.shape == (16, N_LABELS)


def test_output_range_is_valid_probability():
    """Sigmoid auf jeder der n_labels Ausgaben -- jeder Wert muss einzeln
    in [0, 1] liegen (kein Softmax, da die Labels nicht exklusiv sind)."""
    model = MLPMultiLabel(input_dim=INPUT_DIM, n_labels=N_LABELS)
    x = torch.randn(16, INPUT_DIM)
    out = model(x)
    assert torch.all(out >= 0) and torch.all(out <= 1)


def test_outputs_are_independent_sigmoid_not_softmax():
    """Die vier Ausgaben dürfen sich nicht zu 1 aufsummieren: Softmax läge
    ebenfalls in [0, 1], würde aber aus Multi-Label ein exklusives
    Mehrklassenproblem machen -- ohne Fehlermeldung."""
    model = MLPMultiLabel(input_dim=INPUT_DIM, n_labels=N_LABELS)
    model.eval()
    out = model(torch.randn(8, INPUT_DIM))
    row_sums = out.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)

