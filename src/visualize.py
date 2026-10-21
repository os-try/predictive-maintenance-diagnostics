"""Hilfsfunktionen für Abbildungen des Projekts (Architekturdiagramme, Streudiagramme, Partial Dependence)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

BLUE = "#3b4cc0"
RED = "#b40426"
TEXT = "#2c3e50"
EDGE = "#cccccc"


def draw_mlp_architecture(hidden_dims, input_labels, output_labels, path, shown_hidden=8):
    """Zeichne das MLP als Knoten-Kanten-Diagramm und speichere es als PNG.

    Parameters
    ----------
    hidden_dims : sequence of int
        Neuronenzahl je Hidden Layer (nur für die Beschriftung).
    input_labels : list of str
        Beschriftung der Eingabeknoten (von oben nach unten).
    output_labels : list of str
        Beschriftung der Ausgabeknoten.
    path : str
        Zieldatei.
    shown_hidden : int, default=8
        Anzahl der tatsächlich gezeichneten Knoten je Hidden Layer; dazwischen
        stehen Auslassungspunkte.
    """
    y_top, y_bottom = 0.88, 0.08
    n_in, n_out = len(input_labels), len(output_labels)
    layer_x = [0.0] + [0.9 + 0.65 * i for i in range(len(hidden_dims))] + [0.9 + 0.65 * len(hidden_dims) + 0.5]
    layer_x = [x / layer_x[-1] for x in layer_x]

    def even(n):
        step = (y_top - y_bottom) / (n - 1)
        return [y_top - i * step for i in range(n)]

    in_y = even(n_in)
    out_y = even(n_out)
    half = shown_hidden // 2
    hid_all = even(shown_hidden + 1)
    hid_y = hid_all[:half] + hid_all[half + 1:]
    dots_y = hid_all[half]

    layers = [in_y] + [hid_y for _ in hidden_dims] + [out_y]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-0.28, 1.12)
    ax.set_ylim(-0.02, 1.0)
    ax.axis("off")

    for li in range(len(layers) - 1):
        for ya in layers[li]:
            for yb in layers[li + 1]:
                ax.plot([layer_x[li], layer_x[li + 1]], [ya, yb], color=EDGE, lw=0.5, alpha=0.8, zorder=1)

    for li, ys in enumerate(layers):
        color = RED if li == len(layers) - 1 else BLUE
        ax.scatter([layer_x[li]] * len(ys), ys, s=420, color=color, zorder=2)

    for x, in_label, y in zip([layer_x[0]] * n_in, input_labels, in_y):
        ax.text(x - 0.03, y, in_label, ha="right", va="center", fontsize=11, color=TEXT)
    for label, y in zip(output_labels, out_y):
        ax.text(layer_x[-1] + 0.03, y, label, ha="left", va="center", fontsize=13, color=TEXT, fontweight="bold")
    for li in range(1, len(hidden_dims) + 1):
        ax.text(layer_x[li], dots_y, "…", ha="center", va="center", fontsize=20, color=TEXT, fontweight="bold")

    ax.text(0.5, 0.985, "MLP-MultiLabel — Architektur", ha="center", va="top", fontsize=15, color=TEXT, fontweight="bold")

    foot = [f"Input\n({n_in} Merkmale)"]
    foot += [f"Hidden {i + 1}\n({h} Neuronen)\nReLU" for i, h in enumerate(hidden_dims)]
    foot += [f"Output\n({n_out} Neuronen)\nSigmoid"]
    for x, t in zip(layer_x, foot):
        ax.text(x, 0.035, t, ha="center", va="top", fontsize=11, color=TEXT)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_tree_icon(ax, cx, base_y, width=0.55, height=0.3):
    """Zeichne ein stilisiertes Baum-Symbol (drei gestapelte Dreiecke mit Stamm)."""
    ax.add_patch(plt.Rectangle((cx - 0.03, base_y - 0.08), 0.06, 0.08, color="#8b5a2b", zorder=3))
    for k, scale in enumerate((1.0, 0.75, 0.5)):
        w = width * scale / 2
        y0 = base_y + k * height * 0.33
        ax.add_patch(plt.Polygon([(cx - w, y0), (cx + w, y0), (cx, y0 + height * 0.55)],
                                 closed=True, facecolor=BLUE, edgecolor="none", alpha=0.85, zorder=4))


def draw_rf_architecture(output_labels, n_trees, n_inputs, path):
    """Zeichne den Aufbau des MultiOutputClassifier (ein Random Forest je Label).

    Parameters
    ----------
    output_labels : list of str
        Ein Wald je Label, von links nach rechts.
    n_trees : int
        Anzahl der Bäume je Wald (nur für die Beschriftung).
    n_inputs : int
        Anzahl der Eingabemerkmale (nur für die Beschriftung).
    path : str
        Zieldatei.
    """
    width, height = 11.0, 6.4
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    ax.text(width / 2, 6.22, "Random Forest MultiLabel — vier unabhängige Modelle (MultiOutputClassifier)",
            ha="center", va="center", fontsize=14, fontweight="bold", color=TEXT)
    ax.text(width / 2, 5.73, f"Dieselben {n_inputs} Eingabemerkmale",
            ha="center", va="center", fontsize=13, fontweight="bold", color=TEXT)

    n = len(output_labels)
    box_w, box_top, box_bottom = 2.06, 4.86, 1.2
    centers = [box_w / 2 + 0.64 + i * (width - box_w - 1.28) / (n - 1) for i in range(n)]
    for cx, label in zip(centers, output_labels):
        ax.annotate("", xy=(cx, 5.0), xytext=(width / 2, 5.49),
                    arrowprops=dict(arrowstyle="->", color="#cccccc", lw=1.5))
        ax.add_patch(plt.Rectangle((cx - box_w / 2, box_bottom), box_w, box_top - box_bottom,
                                   facecolor="#f3f4f6", edgecolor=BLUE, linewidth=2, zorder=2))
        ax.text(cx, 4.38, f"Random Forest\nfür {label}", ha="center", va="center",
                fontsize=13, fontweight="bold", color=TEXT, linespacing=1.3)
        ax.text(cx, 3.82, f"({n_trees} Bäume)", ha="center", va="center", fontsize=11, color=TEXT)
        for base_y in (3.03, 2.2):
            for dx in (-0.54, 0.0, 0.54):
                _draw_tree_icon(ax, cx + dx, base_y)
        ax.text(cx, 1.62, "Mittelwert der\nBaum-Wahrscheinlichkeiten", ha="center", va="center",
                fontsize=8, style="italic", color=TEXT, linespacing=1.3)
        ax.annotate("", xy=(cx, 0.62), xytext=(cx, 1.14),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2))
        ax.text(cx, 0.38, f"P({label})", ha="center", va="center", fontsize=14,
                fontweight="bold", color=RED)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_partial_dependence(estimators, X, target_columns, feature_columns, numeric_columns, scaler, path,
                            min_importance=0.15, max_features=2, grid_resolution=60, font_scale=1.0):
    """Zeichne Partial-Dependence-Kurven der wichtigsten Merkmale je Fehlermodus.

    Je Fehlermodus werden bis zu ``max_features`` numerische Merkmale mit einer
    Feature Importance von mindestens ``min_importance`` gezeigt. Die Panels eines
    Modus teilen sich die y-Achse; die x-Achse ist in Originaleinheiten.

    Parameters
    ----------
    estimators : sequence of fitted RandomForestClassifier
        Ein Modell je Fehlermodus (z.B. ``MultiOutputClassifier.estimators_``).
    X : pandas.DataFrame
        Standardisierte Eingabedaten mit den Spalten ``feature_columns``.
    target_columns : sequence of str
        Namen der Fehlermodi in derselben Reihenfolge wie ``estimators``.
    feature_columns : sequence of str
        Alle Eingabespalten des Modells (Reihenfolge wie beim Training).
    numeric_columns : sequence of str
        Standardisierte numerische Spalten (Reihenfolge wie beim Anpassen des Scalers).
    scaler : sklearn.preprocessing.StandardScaler
        Zum Zurückrechnen der x-Achse in Originaleinheiten.
    path : str or Path
        Zielpfad der PNG-Datei.
    font_scale : float, default=1.0
        Skalierungsfaktor für alle Schriftgrößen (z. B. 1.4 für die Präsentation).
    """
    mean = dict(zip(numeric_columns, scaler.mean_))
    scale = dict(zip(numeric_columns, scaler.scale_))
    groups = []
    for label, est in zip(target_columns, estimators):
        importance = pd.Series(est.feature_importances_, index=feature_columns).sort_values(ascending=False)
        chosen = [f for f in importance.index if f in numeric_columns and importance[f] >= min_importance][:max_features]
        curves = [partial_dependence(est, X[list(feature_columns)], [list(feature_columns).index(f)],
                                     grid_resolution=grid_resolution, percentiles=(0.001, 0.999), kind="average")
                  for f in chosen]
        groups.append((label, chosen, importance, curves))

    n_panels = sum(len(g[1]) for g in groups)
    n_cols = 4
    n_rows = -(-n_panels // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    k = 0
    for label, chosen, importance, curves in groups:
        y_max = max(c["average"][0].max() for c in curves) * 100 * 1.08
        for feature, curve in zip(chosen, curves):
            ax = axes[k]
            x = curve["grid_values"][0] * scale[feature] + mean[feature]
            ax.plot(x, curve["average"][0] * 100, color=RED, lw=2)
            ax.set_ylim(0, y_max)
            observed = X[feature] * scale[feature] + mean[feature]
            ax.plot(observed, np.full(len(observed), y_max * 0.015), "|", color=BLUE, alpha=0.12, ms=8)
            ax.set_title(f"{label}: {feature}\n(Importance {importance[feature]:.2f})", fontsize=9 * font_scale, fontweight="bold")
            ax.set_xlabel(feature, fontsize=9 * font_scale)
            if k % n_cols == 0:
                ax.set_ylabel("Ausfallwahrscheinlichkeit [%]", fontsize=9 * font_scale)
            ax.tick_params(labelsize=8 * font_scale)
            ax.grid(alpha=0.25)
            k += 1
    for ax in axes[k:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_failure_scatter(df, target_columns, panels, path, grid_columns=2):
    """Zeichne je Fehlermodus ein Streudiagramm zweier Merkmale und speichere es als PNG.

    Je Panel erscheinen die Beobachtungen des betrachteten Modus in Rot, Beobachtungen
    ohne jeden Ausfall in Grau (halbtransparent) und Beobachtungen, die nur durch einen
    der übrigen Modi ausfallen, in Blau. Liegt bei einer Beobachtung der betrachtete
    Modus zusammen mit einem anderen vor, zählt sie als Rot.

    Parameters
    ----------
    df : pandas.DataFrame
        Daten mit allen Merkmalen und den Zielspalten (z.B. der Trainingssplit).
    target_columns : sequence of str
        Die betrachteten Fehlermodi; RNF ist nicht enthalten und zählt als "kein Ausfall".
    panels : sequence of (str, str, str)
        Je Panel (Fehlermodus, x-Merkmal, y-Merkmal). Die Legende steht einmal
        unterhalb der Panels, damit sie keine Punkte verdeckt.
    path : str or Path
        Zielpfad der PNG-Datei.
    grid_columns : int, default=2
        Anzahl der Panel-Spalten.
    """
    grey = "#b8b8b8"
    any_failure = df[list(target_columns)].sum(axis=1) > 0
    n_rows = -(-len(panels) // grid_columns)
    fig, axes = plt.subplots(n_rows, grid_columns, figsize=(5.5 * grid_columns, 4 * n_rows))
    for ax, (mode, x, y) in zip(np.atleast_1d(axes).ravel(), panels):
        healthy = df[~any_failure]
        this = df[df[mode] == 1]
        other = df[any_failure & (df[mode] == 0)]
        ax.scatter(healthy[x], healthy[y], s=6, color=grey, alpha=0.25, lw=0, label="kein Ausfall")
        ax.scatter(other[x], other[y], s=14, color=BLUE, alpha=0.8, lw=0, label="Ausfall durch anderen Modus")
        ax.scatter(this[x], this[y], s=22, color=RED, alpha=0.95, lw=0, label="Ausfall durch den Modus im Titel")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{mode} (n={len(this)}; anderer Modus: n={len(other)})", fontweight="bold")
        ax.grid(alpha=0.25)
    handles, labels = np.atleast_1d(axes).ravel()[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False, markerscale=1.8,
               bbox_to_anchor=(0.5, -0.03))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_label_learning_curves(history, target_columns, path):
    """Zeichne die Validierungs-PR-AUC aller Labels über die Epochen in einer Grafik.

    Parameters
    ----------
    history : dict
        Rückgabe von :func:`src.train.train_classifier` (benötigt
        ``val_pr_auc`` und ``val_pr_auc_mean``).
    target_columns : list of str
        Namen der Labels in der Spaltenreihenfolge von ``val_pr_auc``.
    path : str
        Zieldatei.

    Je Label eine farbige Kurve (``coolwarm``-Töne), dazu der Mittelwert
    (grau gepunktet) und die gewählte Epoche (senkrechte Linie). Die
    Legende steht unterhalb der Zeichenfläche.
    """
    per_label = np.array(history["val_pr_auc"], dtype=float)
    mean_curve = np.array(history["val_pr_auc_mean"], dtype=float)
    best_epoch = int(np.argmax(mean_curve))  # erste Epoche mit höchstem Mittelwert
    colors = ["#b40426", "#f08a6c", "#7b9ff9", "#3b4cc0"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, name in enumerate(target_columns):
        ax.plot(per_label[:, i], color=colors[i % len(colors)], lw=1.8, label=name)
    ax.plot(mean_curve, color="#555555", ls=":", lw=2, label="Mittelwert (4 Labels)")
    ax.axvline(best_epoch, color="black", ls="--", lw=1, label="gewählte Epoche")
    ax.set_xlabel("Epoche")
    ax.set_ylabel("Validierungs-PR-AUC")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=6, frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
