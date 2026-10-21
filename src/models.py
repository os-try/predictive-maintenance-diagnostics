"""Modelldefinition für das Projekt.

Ausführliche Begründungen: siehe docs/implementation_notes.md.
"""
import torch.nn as nn

# Ergebnis der Architektursuche (Notebook 02), Begründung: docs/implementation_notes.md
HIDDEN_DIMS = (512,)


class MLPMultiLabel(nn.Module):
    """Feedforward-Netz für Multi-Label-Klassifikation.

    Feedforward-Netz mit ReLU-Aktivierung. Tiefe und Breite (eine Hidden Layer
    mit 512 Neuronen) wurden in einer Gittersuche über 1--3 Schichten und
    Breiten von 32 bis 512 mit je fünf Seeds bestimmt, nicht unreflektiert
    übernommen -- siehe docs/implementation_notes.md für den Vergleich.

    Parameters
    ----------
    input_dim : int
        Anzahl der Eingabemerkmale.
    n_labels : int, default=4
        Anzahl der gleichzeitig vorherzusagenden Zielspalten (TWF, HDF,
        PWF, OSF).
    hidden_dims : tuple of int, default=HIDDEN_DIMS

    Notes
    -----
    Sigmoid wird elementweise auf alle ``n_labels`` Ausgänge angewendet
    (nicht Softmax) -- die vier Fehlermodi sind laut Datensatzdefinition
    unabhängig und können gleichzeitig auftreten (Mehrfachlabel, kein
    exklusives Mehrklassenproblem).
    """

    def __init__(self, input_dim: int, n_labels: int = 4, hidden_dims=HIDDEN_DIMS):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, n_labels))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Vorwärtsdurchlauf.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor of shape (batch_size, n_labels)
            Ausfallwahrscheinlichkeit je Fehlermodus.
        """
        return self.net(x)
