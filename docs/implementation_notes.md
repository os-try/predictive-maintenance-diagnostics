# Implementierungs-Notizen

Ausführliche Begründungen für Design-Entscheidungen, die absichtlich nicht in
den Funktions-Docstrings selbst stehen (sklearn-Stil im Code, ausführliche
Erklärungen hier).

## Warum RNF ausgeschlossen wird

Laut Datensatzbeschreibung hat jeder Prozess unabhängig von seinen Parametern
eine konstante Ausfallwahrscheinlichkeit von 0,1 % für RNF. Das unterscheidet
sich fundamental von den anderen vier Modi: TWF, HDF, PWF und OSF sind alle an
konkrete, messbare Schwellenbedingungen auf den Prozessparametern geknüpft
(Zeitfenster, Temperaturdifferenz, Leistung, Verschleiß×Drehmoment). RNF hat
keine solche Bedingung — es ist von Konstruktion aus statistisch unabhängig
von allen verfügbaren Merkmalen. 

## Warum Multi-Label statt Mehrklassen-Klassifikation

Getestet in `01_data_pipeline.ipynb`: 22 Beobachtungen im Datensatz haben
gleichzeitig zwei der vier betrachteten Modi aktiv, eine sogar drei. Ein
exklusives Mehrklassenproblem (eine Klasse pro Beobachtung, Softmax-Ausgabe)
würde diese Ko-Okkurrenz nicht abbilden können. Beide Modelle nutzen deshalb
für jedes Label eine unabhängige, nicht auf 1 normierte Sigmoid-Ausgabe.

## `src/data.py`

### Echte Multi-Label-Stratifizierung (`MultilabelStratifiedShuffleSplit`)

`split_data()` stratifizierte zunächst nur nach `Machine failure` (dem
übergeordneten Binärlabel), nicht nach allen vier Zielspalten gleichzeitig —
eine Methode, die für eine einzelne Zielspalte korrekt, für vier gleichzeitige
Zielspalten aber unzureichend ist. Ein direkter empirischer Vergleich zeigte: Bei reiner
Binärlabel-Stratifizierung wich der Testset-Anteil einzelner Modi um bis zu
0,25 Prozentpunkte vom Gesamtanteil ab (z. B. HDF: 1,40 % statt 1,15 %), bei
TWF führte das sogar zu nur 5 statt der proportional erwarteten 7 Testfällen
— ausgerechnet bei dem ohnehin am schlechtesten vorhersagbaren,
fallzahlärmsten Modus.

Mit `MultilabelStratifiedShuffleSplit` aus dem Paket `iterative-stratification`
(simultane Stratifizierung über alle vier Zielspalten) sinkt die maximale
Abweichung auf durchweg unter 0,02 Prozentpunkte — eine 5- bis 15-fache
Verbesserung gegenüber der vorherigen Methode. Die Funktion ruft den Split
zweimal auf (erst Test von Train+Val trennen, dann Train von Val), jeweils
mit der vollen `(N, 4)`-Zielmatrix statt einer einzelnen Spalte.

### `MultiLabelDataset`

`self.y` hat die Form (N, 4) statt (N, 1) — eine Zeile pro Beobachtung, eine Spalte pro Fehlermodus, statt
einer einzelnen kombinierten Zielspalte.

## `src/visualize.py`

### `draw_failure_scatter()` -- Streudiagramme je Fehlermodus (Notebook 01)

Je Fehlermodus ein Streudiagramm zweier Merkmale, nur auf dem Trainingssplit
(das Testset bleibt auch für die Visualisierung unberührt). Rot: Beobachtungen des
Modus (auch wenn zugleich ein anderer Modus vorliegt); grau: kein Ausfall (ein
reiner RNF-Ausfall zählt als "kein Ausfall", da RNF ausgeschlossen ist); blau:
Ausfall nur durch einen der übrigen Modi, als Kontrolle, dass sich der Modus von
den anderen unterscheiden lässt. Die Merkmalspaare entsprechen den
Fehlerbedingungen (HDF: Temperaturdifferenz/Drehzahl, PWF: Drehzahl/Drehmoment,
OSF und TWF: Verschleiß/Drehmoment).

### `draw_label_learning_curves()` -- Validierungs-PR-AUC je Label (Notebook 02)

`train_classifier()` protokolliert zusätzlich zur gemittelten Validierungs-PR-AUC die PR-AUC jedes einzelnen Labels je Epoche (`history["val_pr_auc"]`, `nan` bei Labels ohne positiven Fall im Validierungsset; der Mittelwert ignoriert diese wie bisher). Early Stopping nutzt weiterhin nur den Mittelwert. Die Zeichenfunktion zeigt alle vier Labels farblich getrennt in einer Grafik, dazu den Mittelwert (grau gepunktet) und die gewählte Epoche; die Kurven liegen auf sehr unterschiedlichem Niveau und bleiben dadurch lesbar. Zweck ist die Diagnose: Der Mittelwert kann unterschiedliche Verläufe verdecken (z. B. TWF, dessen Kurve kaum steigt). Einschränkung: Das Validierungsset enthält je Label nur 7 bis 17 positive Fälle, kleine Ausschläge der Einzelkurven sind daher Rauschen.

## `src/models.py`

### `MLPMultiLabel` — Tiefe und Breite per Gittersuche bestimmt (512)

Die Architektur war zunächst mit 32/16 Neuronen gesetzt, ohne für die
Multi-Label-Aufgabe geprüft zu sein. Ein erster Breitenvergleich mit je einem Lauf (32/16, 64/32, 128/64)
sprach für mehr Kapazität, sagt aber wegen der Seed-Streuung wenig über die
Tiefe und nichts über die Stabilität aus. Deshalb wurde ein Gitter über
1 bis 3 Hidden Layers und erste Breiten von 32 bis 512 Neuronen untersucht
(jede weitere Schicht halbiert die Breite), je Architektur mit fünf Seeds
(`search_mlp_architecture()` in `src/train.py`, Ergebnis
`results/mlp/mlp_architecture_search.csv`). Alle anderen Hyperparameter
(Adam, lr=1e-3, Batch 64, gewichtete BCE, Early Stopping mit Geduld 15)
blieben gleich.

Bewertet wird die Validierungs-PR-AUC des Modells mit der besten Epoche,
gemittelt über HDF, PWF und OSF. TWF bleibt bei der Auswahl außen vor: Seine
PR-AUC liegt architekturunabhängig bei 0,07 bis 0,13 und schwankt rein zufällig
(Zufallskomponente der Datengenerierung); würde sie in den Mittelwert
eingehen, entschiede Rauschen über die Rangfolge. Das Testset wird bei der
Auswahl nicht verwendet.

| Architektur | Parameter | Val-PR-AUC ohne TWF (Mittel ± Std, 5 Seeds) | Val-PR-AUC TWF |
|---|---|---|---|
| 32 | 484 | 0,879 ± 0,016 | 0,090 |
| 64 | 964 | 0,929 ± 0,012 | 0,081 |
| 128 | 1.924 | 0,918 ± 0,016 | 0,090 |
| 256 | 3.844 | 0,930 ± 0,009 | 0,073 |
| 512 | 7.684 | 0,936 ± 0,004 | 0,072 |
| 32/16 | 948 | 0,867 ± 0,032 | 0,100 |
| 64/32 | 2.916 | 0,901 ± 0,027 | 0,084 |
| 128/64 | 9.924 | 0,920 ± 0,022 | 0,078 |
| 256/128 | 36.228 | 0,933 ± 0,012 | 0,088 |
| 512/256 | 137.988 | 0,932 ± 0,006 | 0,080 |
| 32/16/8 | 1.052 | 0,859 ± 0,037 | 0,098 |
| 64/32/16 | 3.380 | 0,894 ± 0,030 | 0,084 |
| 128/64/32 | 11.876 | 0,912 ± 0,013 | 0,085 |
| 256/128/64 | 44.228 | 0,925 ± 0,024 | 0,107 |
| 512/256/128 | 170.372 | 0,915 ± 0,008 | 0,130 |

Ergebnis: **512** (ein Hidden Layer, 7.684 Parameter) hat den besten
Mittelwert (0,936) und die kleinste Streuung der Spitzengruppe
(± 0,004; schlechtester Seed 0,930). 256/128 (0,933 ± 0,012; schlechtester
Seed 0,912) und 512/256 (0,932 ± 0,006) liegen im Rauschen: Der Abstand
512 gegenüber 256/128 beträgt 0,003 (Welch-t-Test p ≈ 0,67). Die Kurve
flacht ab etwa 256 Neuronen ab, eine dritte Schicht ist in keiner Breite
besser (bestes Dreischichtnetz 256/128/64: 0,925 ± 0,024).

Weil die Mittelwerte der Spitzengruppe statistisch nicht trennbar sind,
wurde vorab festgelegt: Es entscheidet erstens die Stabilität über die
Seeds, zweitens (bei Gleichstand) die kleinere Variante. Beides spricht für
512: kleinste Streuung der Spitzengruppe bei höchstem Mittel, und mit 7.684
Parametern rund fünfmal kleiner als 256/128 (36.228) und 18-mal kleiner als
512/256 (137.988). Ein Overfitting-Maß (Trainings- vs. Validierungs-PR-AUC)
liegt in den Suchergebnissen nicht vor; als Indiz dient allein die
Seed-Streuung. Gewählt wird 512 als Standardwert. Der Wert steht einmal als Konstante `HIDDEN_DIMS` in `src/models.py`; Modell, Architekturdiagramm und App lesen ihn von dort. Ändert sich das Ergebnis der Suche, wird nur diese Konstante angepasst.

Zusätzlich wurde geprüft, ob das Early-Stopping-Kriterium (gemittelte
Validierungs-PR-AUC über alle vier Labels) TWF systematisch benachteiligt --
empirisch nicht: TWFs PR-AUC blieb über das gesamte Training flach (zwischen
etwa 0,05 und 0,12, abgelesen aus `mlp_learning_curves_per_label.png`). Die
gewählte Epoche liegt bei etwa 0,085, das TWF-Maximum bei etwa 0,12.
Eine andere Epoche würde TWF nicht nennenswert verbessern;
das Kriterium musste daher nicht geändert werden.

Die Suche läuft in Notebook 02 und dauert auf einer CPU etwa 30 Minuten;
sie schreibt `results/mlp/mlp_architecture_search.csv` und die Heatmap.

### Wirkungsrichtung der Merkmale: Partial Dependence (Notebook 05)

Die Feature Importances des Random Forest geben nur die Stärke eines Merkmals
an, nicht die Richtung. `draw_partial_dependence()` in `src/visualize.py` berechnet
mit `sklearn.inspection.partial_dependence` je Einzelmodell des
`MultiOutputClassifier` die durchschnittliche Ausfallwahrscheinlichkeit in
Abhängigkeit vom Merkmalswert. Je Fehlermodus werden die numerischen Merkmale
mit Importance >= 0,15 gezeigt (höchstens zwei), die Panels eines Modus teilen
die y-Achse, die x-Achse ist in Originaleinheiten (Rückrechnung über den
Scaler). Die extremsten 0,1 % der Werte werden abgeschnitten (Perzentile
0,001/0,999), damit seltene Randbereiche wie PWF unter 3 500 W sichtbar bleiben.

Gewählt wurde Partial Dependence statt eines SHAP-Beeswarms, weil die Kurven
die Schwellen der Fehlerbedingungen direkt zeigen und ohne zusätzliche
Bibliothek auskommen. Einschränkungen: Die Kurve mittelt über die übrigen
Merkmale (bei HDF müssen zwei Bedingungen zugleich erfüllt sein, die einzelne
Kurve erreicht daher nur etwa 10 %), und sie nimmt unabhängige Merkmale an
(Leistung, Drehmoment und Drehzahl hängen zusammen). Der MLP wird nicht
erklärt. Bei TWF zeigt die Kurve nur, dass Tool wear dominiert, nicht welche
Beobachtung ausfällt -- die Zufallskomponente bleibt unsichtbar.

## `src/train.py`

### `compute_pos_weight()` liefert einen Vektor

Jede der vier Zielspalten hat eine eigene, stark unterschiedliche
Ausfallhäufigkeit (TWF: 46, HDF: 115, PWF: 95, OSF: 98 Fälle im
Gesamtdatensatz von 10.000). Ein einzelnes gemeinsames Klassengewicht würde
die selteneren Modi systematisch benachteiligen.
Die Funktion berechnet deshalb ein separates `n_negative/n_positive`-Gewicht
je Spalte. Ein `torch.clamp(min=1)` schützt gegen Division durch 0, falls ein
Label im Trainingssplit zufällig keine einzige positive Beobachtung enthält
(bei TWF mit insgesamt nur ~32 Trainingsfällen theoretisch nicht
ausgeschlossen, auch wenn es im tatsächlichen Split nicht eingetreten ist).

### Warum `MultiOutputClassifier` statt des nativen Multi-Output-Random-Forest

`RandomForestClassifier` unterstützt mehrspaltiges `y` nativ, ohne Wrapper.
Der entscheidende Nachteil dafür: `feature_importances_` ist dann ein
einzelner, über alle Ausgabespalten gemittelter Vektor — es lässt sich damit
nicht mehr unterscheiden, welches Merkmal für welchen Fehlermodus relevant
ist. Das widerspricht dem Kernanliegen dieses Projekts (Diagnostik: *welcher*
Mechanismus liegt vor, warum). `MultiOutputClassifier` trainiert stattdessen
vier vollständig unabhängige `RandomForestClassifier`-Modelle (eines je
Label), jedes mit eigener `feature_importances_`. Kostenpunkt: etwa die
vierfache Trainingszeit gegenüber dem nativen Ansatz — bei dieser
Datenmenge (10.000 Zeilen) nicht spürbar ins Gewicht fallend.

### `evaluate_classifier()` / `evaluate_sklearn_classifier()` geben ein DataFrame zurück

Eine flache Rückgabe mit einem Skalarwert je Metrik würde nicht ausreichen.
Stattdessen wird ein `pandas.DataFrame` zurückgegeben, eine Zeile je
Zielspalte — notwendig, weil es nicht mehr "die eine" Precision/Recall/F1 gibt,
sondern vier davon, die inhaltlich stark unterschiedlich ausfallen
(siehe README, Abschnitt "Optimierung", Ergebnistabelle).

### `evaluate_sklearn_classifier()` und die Struktur von `predict_proba()`

Sowohl beim nativen Multi-Output-`RandomForestClassifier` als auch bei
`MultiOutputClassifier` gibt `predict_proba()` eine **Liste** von Arrays
zurück (eines je Ausgabespalte), nicht ein einzelnes zusammenhängendes Array
wie bei einem gewöhnlichen Klassifikator. Jedes Element der Liste hat die
Form `(n_samples, 2)` für `[P(Klasse=0), P(Klasse=1)]`. Die Funktion führt
diese Liste zu einem einzigen `(n_samples, n_labels)`-Array aus den
`P(Klasse=1)`-Werten zusammen, um dieselbe Struktur wie bei
`evaluate_classifier()` (PyTorch-Modelle) zu erhalten.

### `find_optimal_thresholds_multilabel()` ruft die Kernfunktion pro Label auf

`find_optimal_threshold()` bestimmt das F1-Maximum entlang der
Precision-Recall-Kurve für eine einzelne Zielspalte und darf nur auf dem
Validierungsset aufgerufen werden (auf dem Testset wäre es Data Leakage). Die
Multi-Label-Variante ruft diese Funktion einmal je
Zielspalte separat auf — es gibt bewusst **einen eigenen Schwellenwert je
Fehlermodus**, nicht einen gemeinsamen. Das ist inhaltlich notwendig: Die
gefundenen optimalen Schwellenwerte unterscheiden sich stark zwischen den
Labels (z. B. TWF: 0,245 vs. PWF: 0,533 beim getunten Random Forest) und
spiegeln die unterschiedliche Schwierigkeit und Klassenbalance jedes
einzelnen Modus wider.

## Hyperparameter-Suche (Notebook 04)

Der `RandomizedSearchCV`-Scorer mittelt die PR-AUC über alle vier Labels
(`mean_pr_auc_multilabel`). Labels ohne positive Fälle im jeweiligen Validierungs-Fold
werden von der Mittelung ausgeschlossen (kein sinnvoller PR-AUC-Wert ohne
positive Beispiele). Es wird ein `PredefinedSplit`
verwendet (Trainingsdaten immer Fold -1, Validierungsdaten immer Fold 0),
nicht klassische k-fache Kreuzvalidierung, um konsistent mit dem bereits
zuvor bestimmten Train/Val/Test-Split zu bleiben.

## TWF am Rand des Verschleißbereichs (App-Beispiel)

Bei 253 min Werkzeugverschleiß (Maximum im Datensatz) liefert die App für TWF: Random Forest 8,6 %, MLP 99,9 %.

- Ab 240 min gibt es nur 11 Zeilen, davon 2 TWF (und 3 OSF). Der Random Forest setzt über den Datenbereich hinaus konstant fort und gibt etwa den dort beobachteten Anteil wieder; er liegt unter seinem TWF-Schwellenwert (rund 0,25).
- Das MLP rechnet mit standardisierten Eingaben und setzt die Tendenz „mehr Verschleiß, mehr Risiko“ fort, bis die Sigmoid-Funktion sättigt. `pos_weight` hebt die Ausgaben insgesamt an; der TWF-Schwellenwert liegt bei 0,958, die 99,9 % liegen nur knapp darüber. Der Wert ist nicht kalibriert.
- TWF hat laut Datensatzdefinition eine Zufallskomponente (Ersatz vs. Ausfall im Fenster 200-240 min). Beide Modelle erkennen auf dem Testset 0 von 7 TWF-Fällen; das MLP ist am Rand empfindlicher, nicht zuverlässiger.
