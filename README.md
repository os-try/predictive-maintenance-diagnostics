# Predictive Maintenance & Diagnostics (4 Targets)

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-15%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

**Multi-Label-Diagnostik von Maschinenausfällen**

Vergleich zweier methodisch unterschiedlicher Modellansätze zur diagnostischen Ausfallvorhersage: ein überwachtes MLP (PyTorch) und ein Random Forest als baumbasiertes Ensemble-Verfahren -- beide erweitert auf vier gleichzeitig vorherzusagende Fehlermodi statt einer einzelnen binären Zielgröße. Ergänzt um Feature Engineering, systematisches Schwellenwert- und Hyperparameter-Tuning sowie separate, je Fehlermodus berechnete Feature Importances.

## Motivation

Statt nur *ob* ein Ausfall bevorsteht, sagen die Modelle in diesem Projekt vorher, *welcher* von vier Fehlermechanismen wahrscheinlich vorliegt -- ein Schritt von reiner Ausfallerkennung in Richtung Diagnostik. Für die praktische Wartungsplanung ist das oft die wertvollere Information ("vermutlich Wärmeableitung, Kühlung prüfen" statt nur "Ausfall wahrscheinlich").

## Datensatz

**AI4I 2020 Predictive Maintenance Dataset** ([Matzka, 2020](https://doi.org/10.1109/AI4I49448.2020.00023)) -- ein synthetischer Fräsprozess mit 10.000 Beobachtungen. Lizenz: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (Namensnennung erforderlich, sonst keine Einschränkungen).

- UCI-Repository (Übersichtsseite): https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Die Rohdatei liegt bereits unter `data/raw/ai4i2020.csv` im Repository (CC BY 4.0 erlaubt die Weitergabe bei Namensnennung). Die Train-/Val-/Test-Splits unter `data/processed/` erzeugt Notebook 01; sie sind ebenfalls versioniert, damit alle Notebooks und Ergebnisse auf denselben Daten beruhen.

Rohmerkmale: `Type` (Produktqualität L/M/H), `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.

Der Datensatz enthält fünf binäre Einzel-Fehlerflags, von denen vier hier separat vorhergesagt werden:

| Kürzel | Bedeutung | Häufigkeit |
|--------|-----------|---|
| TWF | Tool Wear Failure -- Werkzeugverschleiß-Ausfall | 46 |
| HDF | Heat Dissipation Failure -- Wärmeableitungs-Ausfall | 115 |
| PWF | Power Failure -- Leistungsausfall | 95 |
| OSF | Overstrain Failure -- Überlastungsausfall | 98 |
| RNF | Random Failures -- Zufallsausfälle ohne erkennbaren Zusammenhang (**ausgeschlossen**, siehe unten) | 19 |

22 Beobachtungen haben zwei der vier betrachteten Modi gleichzeitig aktiv, eine sogar drei -- echtes Multi-Label, kein exklusives Mehrklassenproblem.

## Feature Engineering

Über die fünf numerischen Rohmerkmale hinaus werden zwei zusätzliche, physikalisch motivierte Merkmale berechnet (`src/data.py`, `add_engineered_features()`):

- **`Power [W]`** -- mechanische Leistung ($P = M \cdot \omega$), aus Drehmoment und Drehzahl berechnet.
- **`Temperature diff [K]`** -- Differenz aus Prozess- und Lufttemperatur, statt zweier stark korrelierter Absolutwerte.

Beide Größen entsprechen -- ohne dass dies bei ihrer Auswahl beabsichtigt war -- nahezu exakt den Bedingungsgrößen zweier der vier Fehlermodi (PWF bzw. HDF). Details siehe Dokumentation, Kapitel 2.5.

## Warum nur vier von fünf Modi

`RNF` wird bewusst ausgeschlossen: Laut Datensatzdefinition hat jeder Prozess eine konstante, von allen Prozessparametern *unabhängige* Ausfallwahrscheinlichkeit von 0,1 %. Das ist kein Stichprobenproblem, sondern strukturell nicht lernbar -- es gibt kein aus den Sensordaten ableitbares Muster, da per Definition keines existiert. Details siehe [`docs/implementation_notes.md`](docs/implementation_notes.md).

## Modelle

1. **MLP-MultiLabel** -- Feedforward-Netz mit einer Hidden Layer (512 Neuronen, per Gittersuche über Tiefe und Breite bestimmt, ReLU), vier unabhängigen Sigmoid-Ausgabeneuronen (eines je Fehlermodus), gewichteter Binary-Cross-Entropy-Loss (ein Klassengewicht je Zielspalte) und Adam-Optimizer.
2. **Random Forest MultiLabel** -- vier separate, unabhängig trainierte Random-Forest-Modelle (`MultiOutputClassifier`, je 300 Bäume), statt eines einzelnen Multi-Output-Modells -- liefert dadurch für jeden Fehlermodus eigene Feature Importances statt einer gemeinsam gemittelten.

Bewertung anhand von Precision, Recall, F1-Score sowie ROC-/PR-AUC je Fehlermodus (Accuracy allein ist wegen des starken Klassenungleichgewichts nicht aussagekräftig).

### Architektursuche (MLP)

Tiefe und Breite des MLP sind nicht gesetzt, sondern gemessen: 15 Architekturen (1 bis 3 Hidden Layers, erste Breite 32 bis 512), je fünf Seeds, bewertet an der Validierungs-PR-AUC über HDF, PWF und OSF (TWF bleibt wegen seiner Zufallskomponente bei der Auswahl außen vor). 512 liegt mit 0,936 ± 0,004 vorn und ist unter den statistisch gleichwertigen Spitzenvarianten (256/128 mit 0,933, 512/256 mit 0,932) die stabilste und kleinste; eine dritte Schicht hilft in keiner Breite. Ergebnis und Heatmap: `figures/mlp/mlp_architecture_search.png` bzw. Notebook 02. Notebook 02 führt die Suche aus (rund 30 Minuten auf einer CPU; im letzten Lauf 31 Minuten) und schreibt die Rohergebnisse nach `results/mlp/mlp_architecture_search.csv`.

## Optimierung

Im Anschluss an den Modellvergleich werden zwei Hebel systematisch optimiert (siehe `04_threshold_and_tuning.ipynb`):

- **Schwellenwert-Optimierung** -- für jedes Modell UND jeden Fehlermodus wird auf dem Validierungsset der F1-optimale Schwellenwert bestimmt (acht Schwellenwerte insgesamt statt eines pauschalen 0,5).
- **Hyperparameter-Suche** (`RandomizedSearchCV`) -- für Random Forest, mit einem eigenen Scorer, der die PR-AUC über alle vier Labels mittelt.

Ergebnis: Random Forest (getunt) erreicht ein F1-Makromittel von **0,909** (ohne den strukturell schwer vorhersagbaren TWF-Modus; das MLP kommt auf 0,802) und gewinnt klar bei HDF und PWF, während das MLP bei OSF vorn liegt (0,897 gegenüber 0,759). TWF bleibt bei beiden Modellen praktisch unbrauchbar: Beide erkennen keinen der 7 Testfälle (F1=0). Details siehe Dokumentation, Kapitel 7.

Ergebnistabelle (Testset, nach Schwellenwert-Optimierung; Random Forest getunt; Quelle: `results/comparison/final_comparison_table.csv`):

| Fehlermodus | MLP F1 | RF F1 | MLP PR-AUC | RF PR-AUC |
|---|---|---|---|---|
| TWF | 0,000 | 0,000 | 0,054 | 0,059 |
| HDF | 0,741 | 0,970 | 0,976 | 1,000 |
| PWF | 0,769 | 1,000 | 0,878 | 1,000 |
| OSF | 0,897 | 0,759 | 0,978 | 0,890 |
| **Makromittel (4 Modi)** | 0,602 | 0,682 | 0,721 | 0,737 |
| **Makromittel (ohne TWF)** | 0,802 | 0,909 | 0,944 | 0,963 |

Das Testset enthält je Modus nur 7 bis 15 Positive; einzelne Unterschiede sind daher mit Vorsicht zu lesen.

### Bemerkenswerter Befund

Die separaten Feature Importances je Label stimmen mit den bekannten Generierungsbedingungen der einzelnen Fehlermodi überein (z. B. HDF: `Temperature diff` + `Rotational speed` dominieren). Dabei ist zu beachten, dass Leistung und Temperaturdifferenz als abgeleitete Merkmale bereitgestellt wurden. Die Schwellen der Bedingungen lernen die Modelle aus den Daten: Ein einzelner, aus dem HDF-Wald extrahierter Baum teilt bei 8,65 K, nahezu exakt am bekannten Schwellenwert von 8,6 K. Partial-Dependence-Kurven (`figures/comparison/rf_partial_dependence.png`) zeigen zusätzlich die Wirkungsrichtung: Bei HDF springt die Ausfallwahrscheinlichkeit unterhalb von etwa 8,6 K Temperaturdifferenz und etwa 1 380 rpm. Details siehe `docs/implementation_notes.md`.

## Projektstruktur

```
predictive-maintenance-diagnostics/
├── .github/
│   └── workflows/
│       └── tests.yml                           # CI: pytest bei jedem Push
├── .gitattributes                              # einheitliche Zeilenenden (LF/CRLF)
├── data/
│   ├── raw/                                    # Originaldaten (AI4I 2020)
│   └── processed/                              # Train-/Val-/Test-Splits
├── docs/
│   └── implementation_notes.md                 # Begründungen zu src/
├── figures/                                    # Grafiken (PNG)
│   ├── eda/
│   │   ├── target_distribution_multilabel.png
│   │   ├── feature_distributions.png
│   │   ├── engineered_feature_distributions.png
│   │   ├── failure_scatter.png                 # Streudiagramme je Modus
│   │   └── correlation_matrix.png
│   ├── mlp/
│   │   ├── mlp_architecture.png
│   │   ├── mlp_architecture_search.png         # Heatmap der Suche
│   │   ├── mlp_learning_curves.png             # Loss, gemittelte PR-AUC
│   │   └── mlp_learning_curves_per_label.png
│   ├── tree/
│   │   ├── rf_architecture.png
│   │   ├── rf_feature_importances_per_label.png
│   │   └── example_tree.png                    # ein Beispielbaum
│   └── comparison/
│       ├── rf_partial_dependence.png
│       ├── rf_partial_dependence_compact.png   # große Schrift
│       ├── actual_vs_predicted_rf.png
│       ├── confusion_matrices_per_label.png
│       └── comparison_f1_per_label.png
├── results/                                    # CSV-Tabellen (aus Notebooks)
│   ├── mlp/                                    # Architektursuche, Testmetriken
│   ├── tree/                                   # RF-Metriken, Importances
│   ├── tuning/                                 # Schwellen, getunter RF
│   └── comparison/                             # final_comparison_table.csv
├── models/
│   ├── mlp_multilabel.pt                       # state_dict
│   ├── random_forest_multilabel.pkl            # ungetunt
│   ├── random_forest_multilabel_tuned.pkl      # von app.py bevorzugt
│   ├── optimal_thresholds_multilabel.json      # Schwellen je Modell/Modus
│   └── scaler.pkl                              # StandardScaler
├── notebooks/
│   ├── 01_data_pipeline.ipynb                  # Daten, EDA, Split
│   ├── 02_mlp_multilabel.ipynb                 # Modell 1: MLP
│   ├── 03_random_forest_multilabel.ipynb       # Modell 2: Random Forest
│   ├── 04_threshold_and_tuning.ipynb           # Schwellen + Tuning
│   └── 05_comparison.ipynb                     # finaler Vergleich
├── src/
│   ├── data.py                                 # Dataset, Split, Skalierung
│   ├── models.py                               # MLPMultiLabel
│   ├── paths.py                                # figure_path(), result_path()
│   ├── train.py                                # Training, Metriken, Suchen
│   └── visualize.py                            # alle Abbildungen
├── tests/
│   ├── conftest.py                             # synthetische Testdaten
│   ├── test_data.py                            # Daten-Integrität
│   ├── test_models.py                          # Form, Wertebereich
│   └── test_train.py                           # Gewichte, Metriken
├── app.py                                      # Streamlit-App
├── style.css                                   # CSS-Theming der App (von app.py geladen)
├── LICENSE                                     # MIT-Lizenz
├── pytest.ini                                  # Suchpfad für src/
├── requirements.txt                            # scikit-learn gepinnt
└── README.md                                   # diese Datei
```

## Setup

**Voraussetzung:** Python ≥ 3.11. `requirements.txt` pinnt `scikit-learn==1.8.0`, weil die dokumentierten Ergebnisse (insbesondere die Zufallssuche in Notebook 04) mit dieser Version erzeugt wurden und gespeicherte `.pkl`-Modelle versionsabhängig sind.

**macOS / Linux (bash/zsh):**
```bash
python3 -m venv .venv-4targets
source .venv-4targets/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv-4targets
.\.venv-4targets\Scripts\Activate.ps1
pip install -r requirements.txt
```

Die Notebooks müssen im Kernel dieser Umgebung laufen (in VS Code/Jupyter den Kernel `.venv-4targets` auswählen); bei `ModuleNotFoundError` das Paket in genau dieser Umgebung nachinstallieren und den Kernel neu starten.

Falls die Aktivierung mit einer Fehlermeldung zur Execution Policy abbricht (PowerShell blockiert das Ausführen von `.ps1`-Skripten standardmäßig), einmalig für den Benutzeraccount erlauben:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Environment wieder deaktivieren** (beide Plattformen):
```bash
deactivate
```

## Notebook-Reihenfolge

```
01_data_pipeline → 02_mlp_multilabel → 03_random_forest_multilabel
→ 04_threshold_and_tuning → 05_comparison
```

Die Nummerierung entspricht direkt der Ausführungsreihenfolge. `01_data_pipeline.ipynb` berechnet die abgeleiteten Merkmale, erzeugt die explorativen Grafiken und die Train-/Val-/Test-Splits, auf denen alle folgenden Notebooks aufbauen.

## Tests

Getestet wird die Infrastruktur rund um Daten und Modelle -- nicht die Vorhersagegüte, die hängt vom Training ab, nicht vom Code. Die Tests laufen mit einem kleinen, synthetischen Datensatz und benötigen daher **nicht** die echten Daten in `data/raw/`.

```bash
pytest
```

(Aus dem Projektordner starten; die Datei `pytest.ini` setzt den Suchpfad für `src/`.) 15 Tests, beschränkt auf das Wesentliche: Datenintegrität (kein Überlapp der Splits, Skalierung nur auf dem Trainingssplit, Stratifizierung je Label, RNF ausgeschlossen, `y`-Form (N, 4)), Division durch Null (Label ohne positive Fälle bei Klassengewicht und Metriken), Konsistenz der Konfusionsmatrix-Zellen, Schwellenwert-Suche sowie Form, Wertebereich und Sigmoid-Charakter der Modellausgabe (Ausgaben summieren sich nicht zu 1) und die Anwendung des jeweils eigenen Schwellenwerts je Label.

`src/` enthält den wiederverwendbaren Code, den sowohl die Notebooks als auch die Tests importieren -- so laufen Notebook-Implementierung und Tests nicht auseinander. Die Docstrings folgen dem knappen, API-fokussierten Stil von Bibliotheken wie scikit-learn (Parameters/Returns/Notes) statt ausführlicher Inline-Kommentare im Funktionscode; die dahinterliegenden Begründungen für Design-Entscheidungen stehen gesammelt in [`docs/implementation_notes.md`](docs/implementation_notes.md).

### Continuous Integration (CI)

Continuous Integration heißt: Bei jeder Änderung läuft automatisch ein Prüflauf, ohne dass jemand daran denken muss. Hier übernimmt das GitHub Actions (`.github/workflows/tests.yml`). Bei jedem `push` und jedem `pull_request` startet GitHub eine frische Ubuntu-Maschine, installiert Python 3.12, PyTorch (CPU-Version) und die Abhängigkeiten aus `requirements.txt` und führt `python -m pytest -q` aus. Schlägt ein Test fehl, wird der Lauf auf GitHub rot markiert. Dadurch fällt ein Fehler (etwa eine Änderung in `src/`, die die Datenaufteilung oder die Schwellenwert-Logik bricht) sofort auf und nicht erst beim nächsten Notebook-Lauf. Die CI prüft nur die 15 Tests; sie trainiert keine Modelle und braucht die echten Daten nicht. Ein sauberer Lauf in einer neu aufgesetzten Umgebung zeigt zugleich, dass `requirements.txt` vollständig ist.

## Web-App

Eine [Streamlit](https://streamlit.io)-Anwendung (`app.py`) macht die Diagnose interaktiv nutzbar -- zeigt für jeden der vier Fehlermodi eine eigene Statuszeile mit individuellem Schwellenwert und einer auf den jeweiligen Mechanismus zugeschnittenen Handlungsempfehlung (z. B. bei HDF: "Kühlung und Drehzahl prüfen").

**Online:** [Streamlit-URL einfügen -- Platzhalter bis zum Deployment](https://PLATZHALTER.streamlit.app)

Weitere Details:
- Modellauswahl per Radiobutton (Random Forest empfohlen, MLP, oder beide nebeneinander).
- Warnt vor der Vorhersage, falls ein Messwert weit außerhalb des typischen AI4I-2020-Trainingsbereichs liegt (Extrapolation, Ergebnis ggf. nicht verlässlich).
- Bei Random Forest werden zusätzlich die einflussreichsten Merkmale je Fehlermodus angezeigt.


Lokal starten:
```bash
streamlit run app.py
```

## Hinweis zu CPU/GPU

Es gibt keine getrennten Notebooks für CPU und GPU -- das Gerät wird im Code automatisch erkannt und einheitlich verwendet:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Beim hier verwendeten, eher kleinen MLP (10.000 Zeilen Tabellendaten) macht eine GPU ohnehin kaum einen spürbaren Unterschied -- CPU-only ist eine gültige Option. Random Forest läuft ohnehin ausschließlich auf der CPU.

## Quellen

- Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications.* AI4I 2020, IEEE.
- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
- Bruce, P., Bruce, A., & Gedeck, P. (2026). *AI-Assisted Statistics for Data Scientists.* Sebastopol, CA: O'Reilly Media.
- Provost, F., & Fawcett, T. (2013). *Data Science for Business.* Sebastopol, CA: O'Reilly Media.
- Géron, A. (2026). *Hands-On Machine Learning with Scikit-Learn and PyTorch.* Santa Rosa, CA: O'Reilly Media.
- Tsoumakas, G., & Katakis, I. (2007). *Multi-Label Classification: An Overview.* International Journal of Data Warehousing and Mining, 3(3), 1–13.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.

## Lizenz

Der Code steht unter der MIT-Lizenz, Volltext in [`LICENSE`](LICENSE). Der Datensatz AI4I 2020 stammt aus dem UCI Machine Learning Repository und unterliegt seiner eigenen Lizenz (CC BY 4.0, Quellenangabe siehe oben).
