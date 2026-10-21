"""Gemeinsame Test-Fixtures für das Projekt.

Synthetische Testdaten (kein Zugriff auf die echten Rohdaten nötig).
"""
import numpy as np
import pandas as pd
import pytest

from src.data import add_engineered_features


@pytest.fixture
def synthetic_df():
    """Kleiner synthetischer Datensatz mit realistischer Struktur.

    200 Beobachtungen, mit absichtlich seltenen positiven Fällen je
    Zielspalte (TWF/HDF/PWF/OSF), analog zum echten Klassenungleichgewicht.
    """
    rng = np.random.default_rng(42)
    n = 200

    df = pd.DataFrame({
        "Type": rng.choice(["L", "M", "H"], size=n, p=[0.5, 0.3, 0.2]),
        "Air temperature [K]": rng.normal(300, 2, n),
        "Process temperature [K]": rng.normal(310, 1, n),
        "Rotational speed [rpm]": rng.normal(1500, 100, n),
        "Torque [Nm]": np.clip(rng.normal(40, 10, n), 0, None),
        "Tool wear [min]": rng.integers(0, 250, n),
    })

    # Vier Zielspalten mit unterschiedlicher, seltener Häufigkeit
    df["TWF"] = 0
    df["HDF"] = 0
    df["PWF"] = 0
    df["OSF"] = 0
    df.loc[rng.choice(n, size=4, replace=False), "TWF"] = 1
    df.loc[rng.choice(n, size=8, replace=False), "HDF"] = 1
    df.loc[rng.choice(n, size=6, replace=False), "PWF"] = 1
    df.loc[rng.choice(n, size=6, replace=False), "OSF"] = 1
    df["RNF"] = 0
    df.loc[rng.choice(n, size=1, replace=False), "RNF"] = 1

    df["Machine failure"] = (
        df[["TWF", "HDF", "PWF", "OSF", "RNF"]].sum(axis=1) > 0
    ).astype(int)

    return add_engineered_features(df)
