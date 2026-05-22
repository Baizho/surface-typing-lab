"""
Word classifier for UniKey-style surface typing inference.

Architecture follows UniKey (UIST 2025) §4.3.1:
  - Input: normalized IKT sequence + L-R one-hot encoding
  - Small feedforward MLP with 4 hidden layers
  - Dropout + ReLU activations
  - Trained with Adam + categorical cross-entropy

With a small dataset (20 reps/word) this serves as a
proof-of-concept pipeline rather than a production system.
"""

import torch
import torch.nn as nn


class WordClassifier(nn.Module):

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)