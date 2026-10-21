"""Trainings- und Evaluationslogik für das Projekt.

Jede Metrik-Funktion arbeitet hier pro Zielspalte (TWF, HDF, PWF, OSF)
statt auf einem einzelnen Ziel -- Rückgabewerte sind entsprechend Listen
bzw. ein DataFrame mit einer Zeile je Label.

Ausführliche Begründungen: siehe docs/implementation_notes.md.
"""
import contextlib
import copy
import io
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import TARGET_COLUMNS


def get_device() -> torch.device:
    """Return ``"cuda"`` if available, else ``"cpu"``."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible training runs."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_pos_weight(train_dataset) -> torch.Tensor:
    """Compute per-label positive-class weights for weighted BCE loss.

    Parameters
    ----------
    train_dataset : MultiLabelDataset
        Muss ein ``.y``-Tensor der Form (N, n_labels) mit Werten 0.0/1.0
        bereitstellen.

    Returns
    -------
    torch.Tensor of shape (n_labels,)
        ``n_negative / n_positive`` je Zielspalte.

    Notes
    -----
    Es wird ein Vektor zurückgegeben (kein einzelner Skalar) --
    jede Zielspalte hat eine eigene, stark unterschiedliche
    Ausfallhäufigkeit und braucht ein eigenes Gewicht (siehe
    docs/implementation_notes.md).
    """
    y = train_dataset.y
    n_pos = y.sum(dim=0)
    n_neg = len(y) - n_pos
    # Schutz gegen Division durch 0, falls ein Label im Trainingssplit
    # zufällig keine einzige positive Beobachtung enthält.
    n_pos_safe = torch.clamp(n_pos, min=1)
    return n_neg / n_pos_safe


class EarlyStopping:
    """Stop training when a validation metric stops improving.

    Arbeitet generisch auf einem einzelnen Skalar-Score (hier: gemittelte
    Validierungs-PR-AUC über alle vier Labels).
    """

    def __init__(self, patience: int = 10, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.best_state = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float, model: nn.Module) -> None:
        is_better = (
            self.best_score is None
            or (self.mode == "max" and score > self.best_score)
            or (self.mode == "min" and score < self.best_score)
        )
        if is_better:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_classifier(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    pos_weight: torch.Tensor = None,
    patience: int = 10,
):
    """Train the Multi-Label-MLP with weighted BCE loss and early stopping.

    Parameters
    ----------
    model : torch.nn.Module
    train_loader, val_loader : torch.utils.data.DataLoader
    device : torch.device
    epochs : int, default=100
    lr : float, default=1e-3
    pos_weight : torch.Tensor of shape (n_labels,), optional
        Siehe :func:`compute_pos_weight`.
    patience : int, default=10
        Early-Stopping-Kriterium: gemittelte Validierungs-PR-AUC über
        alle vier Labels (siehe :class:`EarlyStopping`).

    Returns
    -------
    model : torch.nn.Module
    history : dict
        Per-epoch ``train_loss``, ``val_loss``, ``val_pr_auc_mean`` sowie
        ``val_pr_auc`` (Liste je Epoche mit einer PR-AUC je Label; ``nan``,
        wenn das Label im Validierungsset keinen positiven Fall hat).
    """
    model.to(device)
    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, mode="max")

    history = {"train_loss": [], "val_loss": [], "val_pr_auc_mean": [], "val_pr_auc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)

            if pos_weight is not None:
                pw = pos_weight.to(device)
                weights = torch.where(y_batch == 1, pw, torch.ones_like(pw))
                loss = (criterion(y_pred, y_batch) * weights).mean()
            else:
                loss = criterion(y_pred, y_batch).mean()

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        all_probs, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch).mean()
                val_losses.append(loss.item())
                all_probs.append(y_pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_targets = np.concatenate(all_targets)

        pr_aucs = []
        for i in range(all_targets.shape[1]):
            if all_targets[:, i].sum() > 0:
                pr_aucs.append(average_precision_score(all_targets[:, i], all_probs[:, i]))
            else:
                pr_aucs.append(float("nan"))
        valid = [v for v in pr_aucs if not np.isnan(v)]
        val_pr_auc_mean = float(np.mean(valid)) if valid else 0.0

        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(np.mean(val_losses))
        history["val_pr_auc_mean"].append(val_pr_auc_mean)
        history["val_pr_auc"].append(pr_aucs)

        early_stopping.step(val_pr_auc_mean, model)
        if early_stopping.should_stop:
            print(f"Early Stopping nach Epoche {epoch} (keine Verbesserung seit {patience} Epochen)")
            break

    model.load_state_dict(early_stopping.best_state)
    return model, history


def _metrics_per_label(y_true: np.ndarray, y_prob: np.ndarray, threshold, target_columns) -> tuple:
    """Gemeinsame Metrik-Berechnung für PyTorch- und sklearn-Multi-Label-Modelle.

    Parameters
    ----------
    y_true, y_prob : numpy.ndarray of shape (N, n_labels)
    threshold : float or array-like of shape (n_labels,)
        Einheitlicher oder je Label unterschiedlicher Schwellenwert.
    target_columns : list of str

    Returns
    -------
    metrics : pandas.DataFrame
        Eine Zeile je Zielspalte mit precision, recall, f1, roc_auc, pr_auc
        sowie den vier Konfusionsmatrix-Zellen (tn, fp, fn, tp) als eigene
        Spalten -- für eine tabellarische Übersicht ohne Grafik.
    confusion_matrices : dict
        ``{label: numpy.ndarray}`` -- die vollständige 2x2-Konfusionsmatrix
        je Label (Zeilen: tatsächliche Klasse, Spalten: vorhergesagte
        Klasse, sklearn-Konvention), für Heatmap-Darstellungen.
    """
    thresholds = np.broadcast_to(threshold, (len(target_columns),))
    rows = []
    matrices = {}
    for i, col in enumerate(target_columns):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        preds = (yp >= thresholds[i]).astype(int)

        cm = confusion_matrix(yt, preds, labels=[0, 1])
        matrices[col] = cm
        tn, fp, fn, tp = cm.ravel()

        rows.append({
            "label": col,
            "threshold": thresholds[i],
            "precision": precision_score(yt, preds, zero_division=0),
            "recall": recall_score(yt, preds, zero_division=0),
            "f1": f1_score(yt, preds, zero_division=0),
            "roc_auc": roc_auc_score(yt, yp) if yt.sum() > 0 else float("nan"),
            "pr_auc": average_precision_score(yt, yp) if yt.sum() > 0 else float("nan"),
            "n_positive": int(yt.sum()),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })
    metrics = pd.DataFrame(rows).set_index("label")
    return metrics, matrices


def evaluate_classifier(model: nn.Module, loader, device: torch.device, threshold=0.5) -> dict:
    """Evaluate a trained Multi-Label PyTorch classifier.

    Parameters
    ----------
    model : torch.nn.Module
    loader : torch.utils.data.DataLoader
    device : torch.device
    threshold : float or array-like of shape (n_labels,), default=0.5

    Returns
    -------
    dict
        Keys: ``metrics`` (pandas.DataFrame, eine Zeile je Label, inkl.
        tn/fp/fn/tp), ``confusion_matrices`` (dict {label: 2x2-array}),
        ``y_true``, ``y_prob`` (beide shape (N, n_labels)).
    """
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            all_probs.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    metrics, confusion_matrices = _metrics_per_label(all_targets, all_probs, threshold, TARGET_COLUMNS)
    return {
        "metrics": metrics,
        "confusion_matrices": confusion_matrices,
        "y_true": all_targets,
        "y_prob": all_probs,
    }


def evaluate_sklearn_classifier(model, X, y, threshold=0.5) -> dict:
    """Evaluate a scikit-learn Multi-Output classifier (Random Forest).

    Parameters
    ----------
    model : object
        Ein mit mehrspaltigem ``y`` trainierter ``RandomForestClassifier``
        (scikit-learn unterstützt Multi-Output nativ, siehe
        docs/implementation_notes.md).
    X : array-like
    y : array-like of shape (N, n_labels)
    threshold : float or array-like of shape (n_labels,), default=0.5

    Returns
    -------
    dict
        Gleiche Struktur wie :func:`evaluate_classifier`.
    """
    # predict_proba() liefert bei Multi-Output eine LISTE von Arrays (eines
    # je Zielspalte), jedes Array shape (N, 2) fuer [P(0), P(1)] -- wird
    # hier zu einem einzigen (N, n_labels)-Array aus P(1)-Werten
    # zusammengefuehrt, um dieselbe Struktur wie bei evaluate_classifier()
    # zu erhalten.
    proba_list = model.predict_proba(X)
    all_probs = np.column_stack([p[:, 1] for p in proba_list])
    all_targets = np.asarray(y)

    metrics, confusion_matrices = _metrics_per_label(all_targets, all_probs, threshold, TARGET_COLUMNS)
    return {
        "metrics": metrics,
        "confusion_matrices": confusion_matrices,
        "y_true": all_targets,
        "y_prob": all_probs,
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    """Find the classification threshold that maximizes F1 for one label.

    Wird pro Zielspalte separat aufgerufen (siehe
    :func:`find_optimal_thresholds_multilabel`).

    Parameters
    ----------
    y_true, y_prob : numpy.ndarray, beide 1-dimensional

    Returns
    -------
    threshold : float
    f1 : float

    Notes
    -----
    Muss auf dem Validierungsset aufgerufen werden, nicht dem Testset
    (Data Leakage sonst) -- siehe docs/implementation_notes.md.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    precisions, recalls = precisions[:-1], recalls[:-1]
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def find_optimal_thresholds_multilabel(y_true: np.ndarray, y_prob: np.ndarray, target_columns=None) -> dict:
    """Find the F1-optimal threshold separately for each label.

    Parameters
    ----------
    y_true, y_prob : numpy.ndarray of shape (N, n_labels)
    target_columns : list of str, optional
        Default: ``TARGET_COLUMNS``.

    Returns
    -------
    dict
        ``{label: (threshold, f1)}`` je Zielspalte.
    """
    if target_columns is None:
        target_columns = TARGET_COLUMNS
    result = {}
    for i, col in enumerate(target_columns):
        result[col] = find_optimal_threshold(y_true[:, i], y_prob[:, i])
    return result


def search_mlp_architecture(
    train_ds,
    val_ds,
    input_dim: int,
    depths=(1, 2, 3),
    widths=(32, 64, 128, 256),
    seeds=(0, 1, 2, 3, 4),
    device=None,
    epochs: int = 150,
    lr: float = 1e-3,
    patience: int = 15,
    batch_size: int = 64,
    target_columns=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Vergleiche MLP-Architekturen (Tiefe x Breite) über mehrere Seeds.

    Die Breite bezeichnet die erste Hidden-Schicht; jede weitere Schicht
    halbiert sie (z. B. Tiefe 3, Breite 128 -> 128/64/32).

    Parameters
    ----------
    train_ds, val_ds : MultiLabelDataset
    input_dim : int
    depths : iterable of int
    widths : iterable of int
    seeds : iterable of int
    device : torch.device, optional
    epochs, lr, patience, batch_size : Trainingshyperparameter wie in
        :func:`train_classifier`.
    target_columns : list of str, optional
        Reihenfolge der Labels; Standard ``TARGET_COLUMNS``.
    verbose : bool, default=True
        Gibt je Lauf eine Fortschrittszeile mit verstrichener Zeit und
        geschätzter Restzeit aus.

    Returns
    -------
    pandas.DataFrame
        Eine Zeile pro (Architektur, Seed) mit Parameterzahl, Epochen und
        Val-PR-AUC je Label sowie gemittelt über alle vier und über die
        drei Labels ohne TWF.
    """
    from torch.utils.data import DataLoader
    from .models import MLPMultiLabel

    if target_columns is None:
        target_columns = TARGET_COLUMNS
    if device is None:
        device = get_device()

    pos_weight = compute_pos_weight(train_ds)
    rows = []
    total_runs = len(depths) * len(widths) * len(seeds)
    start = time.time()
    for depth in depths:
        for width in widths:
            hidden = tuple(max(width // (2 ** k), 4) for k in range(depth))
            for seed in seeds:
                set_seed(seed)
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
                model = MLPMultiLabel(input_dim, len(target_columns), hidden_dims=hidden)
                n_params = sum(p.numel() for p in model.parameters())
                with contextlib.redirect_stdout(io.StringIO()):
                    model, history = train_classifier(
                        model, train_loader, val_loader, device,
                        epochs=epochs, lr=lr, pos_weight=pos_weight, patience=patience,
                    )
                res = evaluate_classifier(model, val_loader, device, threshold=0.5)
                pr = [
                    average_precision_score(res["y_true"][:, i], res["y_prob"][:, i])
                    for i in range(len(target_columns))
                ]
                row = {
                    "depth": depth, "width": width,
                    "architecture": "/".join(str(h) for h in hidden),
                    "seed": seed, "n_params": n_params,
                    "epochs": len(history["train_loss"]),
                }
                row.update({f"pr_auc_{c}": v for c, v in zip(target_columns, pr)})
                row["pr_auc_mean4"] = float(np.mean(pr))
                row["pr_auc_mean_noTWF"] = float(np.mean(pr[1:]))
                rows.append(row)
                if verbose:
                    done = len(rows)
                    elapsed = time.time() - start
                    eta = elapsed / done * (total_runs - done)
                    print(
                        f"[{done}/{total_runs}] {row['architecture']:<12} Seed {seed}: "
                        f"Val-PR-AUC (ohne TWF) {row['pr_auc_mean_noTWF']:.3f} | "
                        f"{elapsed / 60:.1f} min vergangen, noch ca. {eta / 60:.1f} min",
                        flush=True,
                    )
    return pd.DataFrame(rows)
