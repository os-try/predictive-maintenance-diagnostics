"""Ablageorte für Grafiken (PNG) und Ergebnistabellen (CSV).

Grafiken liegen unter ``figures/<thema>/``, Tabellen unter ``results/<thema>/``.
Die Themenordner sind ``eda``, ``mlp``, ``tree``, ``tuning`` und ``comparison``.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"


def _in_topic(base: Path, topic: str, name: str) -> Path:
    path = base / topic / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def figure_path(topic: str, name: str) -> Path:
    """Pfad zu einer Grafik in ``figures/<topic>/``; legt den Ordner bei Bedarf an."""
    return _in_topic(FIGURES_DIR, topic, name)


def result_path(topic: str, name: str) -> Path:
    """Pfad zu einer Tabelle in ``results/<topic>/``; legt den Ordner bei Bedarf an."""
    return _in_topic(RESULTS_DIR, topic, name)
