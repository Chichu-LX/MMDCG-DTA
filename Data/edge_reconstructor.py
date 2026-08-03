"""Three-class differentiable contact scorer used in Stages 2 and 3."""

import torch
import torch.nn as nn


class EdgeReconstructor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, ligand_features, protein_features, distances):
        inputs = torch.cat((ligand_features, protein_features, distances), dim=-1)
        return self.network(inputs)
