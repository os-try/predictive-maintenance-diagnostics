"""Datenaufbereitung für das 4-Targets-Projekt (Multi-Label-Diagnostik).

Statt der kombinierten Zielgröße ``Machine failure`` werden vier der fünf
Einzel-Fehlermodi als separate, gleichzeitig aktive Ziele behandelt (RNF
ausgeschlossen, siehe docs/implementation_notes.md).

Ausführliche Begründungen: siehe docs/implementation_notes.md.
"""
import math

import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

RAW_FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
ENGINEERED_FEATURE_COLUMNS = ["Power [W]", "Temperature diff [K]"]
FEATURE_COLUMNS = RAW_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS

CATEGORICAL_COLUMN = "Type"
ENCODED_TYPE_COLUMNS = ["type_H", "type_L", "type_M"]

# Die vier mechanistisch erklärbaren Fehlermodi (RNF bewusst ausgeschlossen,
# da laut Datensatzdefinition unabhängig von allen Prozessparametern).
TARGET_COLUMNS = ["TWF", "HDF", "PWF", "OSF"]
EXCLUDED_TARGET = "RNF"
BINARY_TARGET_COLUMN = "Machine failure"  # nur für explorative Grafiken genutzt (Einfärbung)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features ``Power [W]`` and ``Temperature diff [K]``.

    Die Merkmalsableitung ist unabhängig von der Zielgröße.
    """
    df = df.copy()
    df["Power [W]"] = (
        df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * math.pi / 60
    )
    df["Temperature diff [K]"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )
    return df


def encode_type(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the ``Type`` column (product quality L/M/H).

    Returns
    -------
    pandas.DataFrame
        `df` mit ``Type`` ersetzt durch ``type_H``/``type_L``/``type_M``
        (dtype ``int``, nicht ``bool``).
    """
    return pd.get_dummies(df, columns=[CATEGORICAL_COLUMN], prefix="type", dtype=int)


def split_data(
    df: pd.DataFrame,
    target_columns=None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    """Multi-label-stratified train/validation/test split (default 70/15/15).

    Parameters
    ----------
    df : pandas.DataFrame
    target_columns : list of str, optional
        Default: ``TARGET_COLUMNS``. Alle vier Spalten werden gleichzeitig
        stratifiziert (``MultilabelStratifiedShuffleSplit``), nicht nur
        das übergeordnete Binärlabel -- siehe docs/implementation_notes.md
        für den empirischen Vergleich beider Ansätze.
    test_size, val_size : float, default=0.15
    random_state : int, default=42

    Returns
    -------
    train, val, test : pandas.DataFrame
    """
    if target_columns is None:
        target_columns = TARGET_COLUMNS

    y = df[target_columns].values
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(msss1.split(df, y))
    train_val, test = df.iloc[train_val_idx], df.iloc[test_idx]

    val_ratio = val_size / (1 - test_size)
    y_tv = train_val[target_columns].values
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(msss2.split(train_val, y_tv))
    train, val = train_val.iloc[train_idx], train_val.iloc[val_idx]

    return train, val, test


def fit_scaler(train_df: pd.DataFrame, columns=FEATURE_COLUMNS) -> StandardScaler:
    """Fit a ``StandardScaler`` on the training split only (avoids leakage)."""
    scaler = StandardScaler()
    scaler.fit(train_df[columns])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler, columns=FEATURE_COLUMNS) -> pd.DataFrame:
    """Apply an already-fitted scaler to any split."""
    df = df.copy()
    df[columns] = scaler.transform(df[columns])
    return df


class MultiLabelDataset(Dataset):
    """PyTorch ``Dataset`` für die Multi-Label-MLP-Variante.

    Parameters
    ----------
    df : pandas.DataFrame
        Bereits encodiert und skaliert.
    feature_columns : list of str, optional
        Default: ``FEATURE_COLUMNS + ENCODED_TYPE_COLUMNS``.
    target_columns : list of str, optional
        Default: ``TARGET_COLUMNS`` (TWF, HDF, PWF, OSF).

    Notes
    -----
    ``y`` hat die Form ``(N, 4)`` statt ``(N, 1)`` -- jede Beobachtung kann mehrere Modi
    gleichzeitig aufweisen (Mehrfachlabel, kein exklusives Mehrklassen-
    problem).
    """

    def __init__(self, df: pd.DataFrame, feature_columns=None, target_columns=None):
        if feature_columns is None:
            feature_columns = FEATURE_COLUMNS + ENCODED_TYPE_COLUMNS
        if target_columns is None:
            target_columns = TARGET_COLUMNS

        self.X = torch.tensor(df[feature_columns].values, dtype=torch.float32)
        self.y = torch.tensor(df[target_columns].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
