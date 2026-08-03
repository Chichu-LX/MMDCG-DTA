"""Stage-2 two-scale dynamic contact reconstruction model."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .MMDCG_DTA_Stage1 import MMDCGDTAModel_Stage1
from .channels import pack_bipartite_features
from .edge_reconstructor import EdgeReconstructor


class MMDCGDTAModel_Stage2(MMDCGDTAModel_Stage1):
    def __init__(self, config):
        super().__init__(config)
        self.atom_edge_reconstructor = EdgeReconstructor(
            self.d_atom_hidden, hidden_dim=64
        )
        self.substructure_edge_reconstructor = EdgeReconstructor(
            self.d_sub_hidden, hidden_dim=64
        )

    @property
    def reconstructors(self):
        return (self.atom_edge_reconstructor, self.substructure_edge_reconstructor)

    def _reconstruct_contacts(self, hierarchy, graph, ligand_hidden, protein_hidden):
        if graph.num_edges() == 0:
            empty_logits = torch.empty(0, 3, device=ligand_hidden.device)
            return torch.empty(0, 1, device=ligand_hidden.device), {
                "logits": empty_logits,
                "stats": {"total": 0, "remove": 0, "keep": 0, "add": 0},
            }

        if hierarchy == "atom":
            reconstructor = self.atom_edge_reconstructor
        elif hierarchy == "substructure":
            reconstructor = self.substructure_edge_reconstructor
        else:
            raise ValueError(f"unknown hierarchy: {hierarchy}")

        # Interaction graphs store ligand nodes before protein nodes within each sample.
        per_graph_nodes = graph.batch_num_nodes().tolist()
        # Counts are recoverable from the explicit side marker and remain differentiable
        # because they are used only to arrange hidden-state slices.
        side_chunks = torch.split(graph.ndata["side"], per_graph_nodes)
        ligand_counts = [int((chunk == 0).sum().item()) for chunk in side_chunks]
        protein_counts = [
            len(chunk) - n_ligand for chunk, n_ligand in zip(side_chunks, ligand_counts)
        ]
        packed = pack_bipartite_features(
            ligand_hidden, protein_hidden, ligand_counts, protein_counts
        )
        src, dst = graph.edges()
        side = graph.ndata["side"]
        ligand_features = torch.where(
            (side[src] == 0).unsqueeze(-1), packed[src], packed[dst]
        )
        protein_features = torch.where(
            (side[src] == 1).unsqueeze(-1), packed[src], packed[dst]
        )
        logits = reconstructor(ligand_features, protein_features, graph.edata["dist"])
        probabilities = F.softmax(logits, dim=-1)
        weights = (probabilities[:, 1] + 2.0 * probabilities[:, 2]).unsqueeze(-1)
        labels = logits.argmax(dim=-1)
        stats = {
            "total": int(labels.numel()),
            "remove": int((labels == 0).sum().item()),
            "keep": int((labels == 1).sum().item()),
            "add": int((labels == 2).sum().item()),
        }
        return weights, {"logits": logits, "stats": stats}

    def forward(self, sample):
        return self._forward_backbone(sample, dynamic_contacts=True)
