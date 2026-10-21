"""Streamlit-App: Diagnostik-Vorhersage.

Statt einer einzelnen Ausfallwahrscheinlichkeit liefert diese App vier
separate Wahrscheinlichkeiten -- eine je Fehlermodus (TWF, HDF, PWF, OSF) --
mit jeweils eigenem, individuell optimiertem Schwellenwert und einer auf den
jeweiligen Mechanismus zugeschnittenen Handlungsempfehlung. Das ist der
eigentliche Mehrwert: statt nur "Ausfall wahrscheinlich" z.B. "vermutlich
Wärmeableitung -- Kühlung prüfen".

RNF (Zufallsausfälle) wird nicht vorhergesagt -- laut Datensatzdefinition
unabhängig von allen Prozessparametern, siehe docs/implementation_notes.md.

Erwartet folgende Dateien relativ zum Projekt-Root:
  - models/mlp_multilabel.pt
  - models/random_forest_multilabel_tuned.pkl  (bevorzugt) bzw.
    models/random_forest_multilabel.pkl        (Fallback)
  - models/scaler.pkl
  - models/optimal_thresholds_multilabel.json

Diese App muss im Projekt-Root liegen (neben src/ und models/).

Start lokal: streamlit run app.py
"""
import json
from pathlib import Path
import pickle

import pandas as pd
import streamlit as st
import torch

from src.data import ENCODED_TYPE_COLUMNS, FEATURE_COLUMNS, RAW_FEATURE_COLUMNS, TARGET_COLUMNS, add_engineered_features
from src.models import MLPMultiLabel

MODELS_DIR = "models"
INPUT_DIM = len(FEATURE_COLUMNS) + len(ENCODED_TYPE_COLUMNS)  # 7 + 3 = 10

# Für jeden Fehlermodus eine kurze, auf den jeweiligen Mechanismus
# zugeschnittene Handlungsempfehlung -- das ist der eigentliche
# Mehrwert dieser App: nicht nur "Ausfall wahrscheinlich", sondern ein
# konkreter Ansatzpunkt für die Instandhaltung.
# Bewusst weiterhin ohne übertriebene Detailtiefe (z.B. keine exakten
# Ersatzteilnummern) -- das würde mehr Sicherheit vortäuschen, als ein
# datenbasiertes Modell tatsächlich bieten kann.
FAILURE_MODE_INFO = {
    "TWF": {
        "name": "Werkzeugverschleiß (Tool Wear Failure)",
        "recommendation": "Unabhängig von dieser Anzeige: Werkzeug planmäßig bei Annäherung "
                           "an das bekannte Risikofenster (200-240 Min Verschleiß) austauschen.",
    },
    "HDF": {
        "name": "Wärmeableitung (Heat Dissipation Failure)",
        "recommendation": "Kühlung und Drehzahl prüfen -- Wärmeableitung eingeschränkt.",
    },
    "PWF": {
        "name": "Leistung (Power Failure)",
        "recommendation": "Drehmoment/Drehzahl-Kombination prüfen -- Leistung außerhalb des Normalbereichs.",
    },
    "OSF": {
        "name": "Überdehnung (Overstrain Failure)",
        "recommendation": "Werkzeugbelastung reduzieren -- Verschleiß und Drehmoment gemeinsam kritisch.",
    },
}

DEFAULT_PROB_THRESHOLD = 0.5


@st.cache_resource
def load_optimal_thresholds() -> dict:
    """Lädt die pro Modell UND pro Fehlermodus individuell optimierten
    Schwellenwerte aus 04_threshold_and_tuning.ipynb.

    Struktur: {modell_name: {label: schwelle}}. Gibt ein leeres dict
    zurück, falls die Datei (noch) nicht existiert.
    """
    path = f"{MODELS_DIR}/optimal_thresholds_multilabel.json"
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def get_threshold(model_key: str, label: str) -> float:
    """Schwellenwert für ein bestimmtes Modell UND einen bestimmten
    Fehlermodus -- es gibt nicht einen, sondern vier Schwellenwerte je
    Modell (siehe docs/implementation_notes.md)."""
    thresholds = load_optimal_thresholds()
    return thresholds.get(model_key, {}).get(label, DEFAULT_PROB_THRESHOLD)


@st.cache_resource
def load_scaler():
    with open(f"{MODELS_DIR}/scaler.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_model(name: str):
    """Lädt das gewählte Multi-Label-Modell.

    Random Forest: bevorzugt die getunte Version (04_threshold_and_tuning.ipynb),
    mit Fallback auf das ungetunte Modell. Beides sind MultiOutputClassifier-
    Objekte (vier separate Random Forests, siehe docs/implementation_notes.md),
    kein einzelnes scikit-learn-Modell wie beim nativen Multi-Output-Ansatz.
    """
    if name == "Random Forest":
        try:
            with open(f"{MODELS_DIR}/random_forest_multilabel_tuned.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            with open(f"{MODELS_DIR}/random_forest_multilabel.pkl", "rb") as f:
                return pickle.load(f)

    model = MLPMultiLabel(input_dim=INPUT_DIM, n_labels=len(TARGET_COLUMNS))
    state = torch.load(f"{MODELS_DIR}/mlp_multilabel.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def build_feature_row(product_type, air_temp, process_temp, rot_speed, torque, tool_wear, scaler):
    """Baue die skalierte Eingabezeile: Rohmerkmale, abgeleitete Merkmale
    (Feature-Engineering-Formeln aus src/data.py) und One-Hot-Type, danach
    derselbe Scaler-Aufruf wie im Training."""
    numeric_df = pd.DataFrame(
        [[air_temp, process_temp, rot_speed, torque, tool_wear]],
        columns=RAW_FEATURE_COLUMNS,
    )
    numeric_df = add_engineered_features(numeric_df)
    numeric_df = numeric_df[FEATURE_COLUMNS]
    numeric_scaled = scaler.transform(numeric_df)

    type_encoded = {col: 0.0 for col in ENCODED_TYPE_COLUMNS}
    type_encoded[f"type_{product_type}"] = 1.0

    return list(numeric_scaled[0]) + [type_encoded[col] for col in ENCODED_TYPE_COLUMNS]


def build_torch_input(*args, **kwargs):
    row = build_feature_row(*args, **kwargs)
    return torch.tensor([row], dtype=torch.float32)


def build_sklearn_input(*args, **kwargs):
    row = build_feature_row(*args, **kwargs)
    all_columns = FEATURE_COLUMNS + ENCODED_TYPE_COLUMNS
    return pd.DataFrame([row], columns=all_columns)


# --- Seiten-Konfiguration ---
st.set_page_config(page_title="Predictive Maintenance Diagnostics", page_icon="🔍")

# CSS-Theming (dunkles Navy/Orange-Schema) liegt in style.css, über mehrere
# Iterationen anhand echter Browser-Entwicklertools-Befunde verfeinert (siehe
# Kommentare dort). Der Toggle-Schalter-Regelblock bleibt wirkungslos, da diese App
# st.radio statt st.toggle nutzt -- schadet aber nicht, da der Selektor dann
# einfach nichts trifft.
_CSS_PATH = Path(__file__).parent / "style.css"
st.markdown(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.title("Predictive Maintenance — Diagnostik")
st.markdown(
    '<p class="app-subtitle">Messwerte eingeben — Wahrscheinlichkeit je Fehlermodus statt nur Ausfall ja/nein.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="app-intro">Diagnose von vier separat vorhergesagten Fehlermechanismen (TWF, HDF, PWF, OSF).</p>',
    unsafe_allow_html=True,
)

# --- Modellauswahl ---
# Standardmäßig stehen BEIDE Modelle direkt nebeneinander zur Auswahl (kein
# separater "Vergleichsmodus"-Toggle): Bei vier gleichzeitigen
# Diagnose-Ausgaben ist ein Modellvergleich Teil des eigentlichen
# Anwendungsfalls dieser App (siehe README, Ergebnistabelle).
st.subheader("Modell")
model_choice = st.radio(
    "Welches Modell soll die Vorhersage liefern?",
    ["Random Forest (empfohlen)", "MLP-MultiLabel", "Beide vergleichen"],
    index=0,
    horizontal=True,
)

# --- Eingabeformular ---
st.subheader("Messwerte")
col1, col2 = st.columns(2)
with col1:
    product_type = st.selectbox(
        "Produktqualität (Type)", ["L", "M", "H"],
        help="Produktqualitätsstufe: L (niedrig), M (mittel) oder H (hoch).",
    )
    air_temp = st.number_input(
        "Air temperature [K]", value=300.0, step=0.1, min_value=0.0,
        help="Umgebungslufttemperatur am Prozess in Kelvin.",
    )
    process_temp = st.number_input(
        "Process temperature [K]", value=310.0, step=0.1, min_value=0.0,
        help="Prozesstemperatur (typisch etwas höher als die Lufttemperatur) in Kelvin.",
    )
with col2:
    rot_speed = st.number_input(
        "Rotational speed [rpm]", value=1500.0, step=1.0, min_value=0.0,
        help="Drehzahl der Spindel in Umdrehungen pro Minute.",
    )
    torque = st.number_input(
        "Torque [Nm]", value=40.0, step=0.1, min_value=0.0,
        help="Am Werkzeug anliegendes Drehmoment in Newtonmeter.",
    )
    tool_wear = st.number_input(
        "Tool wear [min]", value=100.0, step=1.0, min_value=0.0,
        help="Bisherige Einsatzzeit des Werkzeugs in Minuten.",
    )

# Großzügig gerundete Wertebereiche der Rohmerkmale im Datensatz; Eingaben
# außerhalb lösen eine Warnung aus.
TRAINING_RANGES = {
    "Air temperature [K]": (290.0, 312.0),
    "Process temperature [K]": (298.0, 322.0),
    "Rotational speed [rpm]": (1000.0, 3000.0),
    "Torque [Nm]": (0.0, 80.0),
    "Tool wear [min]": (0.0, 260.0),
}


def check_out_of_range(values: dict) -> list:
    warnings = []
    for feature_name, value in values.items():
        low, high = TRAINING_RANGES[feature_name]
        if value < low or value > high:
            warnings.append(f"{feature_name} = {value:g} (üblicher Trainingsbereich: {low:g}–{high:g})")
    return warnings


def predict_multilabel(model_name: str, x_torch, x_sklearn) -> dict:
    """Liefert {label: wahrscheinlichkeit} für ein einzelnes Modell.

    Kapselt den Unterschied zwischen PyTorch- und scikit-learn-Inferenz an
    einer Stelle, damit die Anzeige-Logik unten unabhängig vom
    Modelltyp ist."""
    if model_name == "Random Forest":
        model = load_model("Random Forest")
        # MultiOutputClassifier.predict_proba() liefert eine Liste von
        # Arrays (eines je Label), siehe docs/implementation_notes.md.
        proba_list = model.predict_proba(x_sklearn)
        return {col: proba_list[i][0, 1] for i, col in enumerate(TARGET_COLUMNS)}
    else:
        model = load_model("MLP-MultiLabel")
        with torch.no_grad():
            probs = model(x_torch)[0]
        return {col: probs[i].item() for i, col in enumerate(TARGET_COLUMNS)}


def status_box(label: str, prob: float, model_key: str):
    """Liefert (Anzeige-Funktion, Text, Alarm?) für einen Fehlermodus."""
    threshold = get_threshold(model_key, label)
    info = FAILURE_MODE_INFO[label]
    message = f"{info['name']}: {prob:.1%}"
    if prob >= threshold:
        return st.error, f"{message} — {info['recommendation']}", True
    if prob >= threshold * 0.5:
        return st.warning, f"{message} — Beobachten, kein akuter Handlungsbedarf.", False
    return st.success, message, False


NO_ALARM_TEXT = "Kein Fehlermodus über seinem individuellen Schwellenwert -- keine akute Diagnose."


def render_predictions(model_name: str, model_key: str, probs: dict):
    """Zeigt die vier Diagnose-Ergebnisse für ein Modell als eigene
    Status-Zeilen -- eine je Fehlermodus, mit individuellem Schwellenwert
    und passender Handlungsempfehlung (siehe FAILURE_MODE_INFO oben)."""
    st.markdown(f"**{model_name}**")
    any_positive = False
    for label in TARGET_COLUMNS:
        show, text, alarm = status_box(label, probs[label], model_key)
        any_positive = any_positive or alarm
        show(text)
    if not any_positive:
        st.caption(NO_ALARM_TEXT)


def render_comparison(rf_probs: dict, mlp_probs: dict):
    """Beide Modelle nebeneinander, ZEILENWEISE je Fehlermodus: Eine lange
    Alarmmeldung (z.B. TWF beim MLP) verschiebt so keine Boxen der anderen
    Spalte; Boxen einer Zeile sind gleich hoch (siehe style.css,
    .st-key-cmp_results)."""
    rf_key, mlp_key = "Random-Forest-MultiLabel-getunt", "MLP-MultiLabel"
    with st.container(key="cmp_results"):
        head_rf, head_mlp = st.columns(2)
        head_rf.markdown("**Random Forest**")
        head_mlp.markdown("**MLP-MultiLabel**")
        any_rf = any_mlp = False
        for label in TARGET_COLUMNS:
            col_rf, col_mlp = st.columns(2)
            show, text, alarm = status_box(label, rf_probs[label], rf_key)
            any_rf = any_rf or alarm
            with col_rf:
                show(text)
            show, text, alarm = status_box(label, mlp_probs[label], mlp_key)
            any_mlp = any_mlp or alarm
            with col_mlp:
                show(text)
        cap_rf, cap_mlp = st.columns(2)
        if not any_rf:
            cap_rf.caption(NO_ALARM_TEXT)
        if not any_mlp:
            cap_mlp.caption(NO_ALARM_TEXT)


predict_clicked = st.button("Diagnose berechnen", use_container_width=True)

if predict_clicked:
    out_of_range = check_out_of_range({
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    })
    if out_of_range:
        st.warning(
            "Mindestens ein Messwert liegt deutlich außerhalb des typischen "
            "Trainingsbereichs -- die Vorhersage ist eine Extrapolation:\n\n"
            + "\n".join(f"- {w}" for w in out_of_range)
        )

    scaler = load_scaler()
    x_torch = build_torch_input(product_type, air_temp, process_temp, rot_speed, torque, tool_wear, scaler)
    x_sklearn = build_sklearn_input(product_type, air_temp, process_temp, rot_speed, torque, tool_wear, scaler)

    if model_choice == "Beide vergleichen":
        rf_probs = predict_multilabel("Random Forest", x_torch, x_sklearn)
        mlp_probs = predict_multilabel("MLP-MultiLabel", x_torch, x_sklearn)
        render_comparison(rf_probs, mlp_probs)
    elif model_choice.startswith("Random Forest"):
        rf_probs = predict_multilabel("Random Forest", x_torch, x_sklearn)
        render_predictions("Random Forest", "Random-Forest-MultiLabel-getunt", rf_probs)

        # Zusätzlicher Interpretierbarkeitsvorteil: separate Feature
        # Importances je Label (dank MultiOutputClassifier, siehe
        # docs/implementation_notes.md) -- zeigt konkret, welche Merkmale
        # für JEDEN der vier Modi ausschlaggebend waren.
        model = load_model("Random Forest")
        feature_names = FEATURE_COLUMNS + ENCODED_TYPE_COLUMNS
        st.markdown("**Einflussreichste Merkmale je Fehlermodus**")
        for i, label in enumerate(TARGET_COLUMNS):
            importances = list(zip(feature_names, model.estimators_[i].feature_importances_))
            importances.sort(key=lambda pair: pair[1], reverse=True)
            top2 = ", ".join(f"{name} ({imp:.0%})" for name, imp in importances[:2])
            st.caption(f"{label}: {top2}")
    else:
        mlp_probs = predict_multilabel("MLP-MultiLabel", x_torch, x_sklearn)
        render_predictions("MLP-MultiLabel", "MLP-MultiLabel", mlp_probs)
