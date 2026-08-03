"""Physics-gated intra- and intermolecular graph-attention channels."""

from __future__ import annotations

import dgl.function as fn
import torch
import torch.nn as nn
from dgl.nn.functional import edge_softmax

from .mechanics import InteractionForceMLP


def _as_counts(values) -> list[int]:
    if torch.is_tensor(values):
        return [int(value) for value in values.detach().cpu().tolist()]
    return [int(value) for value in values]


def pack_bipartite_features(h_ligand, h_protein, ligand_counts, protein_counts):
    """Match DGL's per-sample ``[ligand, protein]`` interaction-node order."""
    ligand_counts = _as_counts(ligand_counts)
    protein_counts = _as_counts(protein_counts)
    ligand_chunks = torch.split(h_ligand, ligand_counts)
    protein_chunks = torch.split(h_protein, protein_counts)
    return torch.cat(
        [part for pair in zip(ligand_chunks, protein_chunks) for part in pair], dim=0
    )


def unpack_bipartite_features(h_all, ligand_counts, protein_counts):
    ligand_counts = _as_counts(ligand_counts)
    protein_counts = _as_counts(protein_counts)
    ligand_parts = []
    protein_parts = []
    offset = 0
    for n_ligand, n_protein in zip(ligand_counts, protein_counts):
        ligand_parts.append(h_all[offset : offset + n_ligand])
        offset += n_ligand
        protein_parts.append(h_all[offset : offset + n_protein])
        offset += n_protein
    return torch.cat(ligand_parts, dim=0), torch.cat(protein_parts, dim=0)


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim=0, negative_slope=0.2):
        super().__init__()
        self.node_projection = nn.Linear(in_dim, out_dim, bias=False)
        self.edge_projection = (
            nn.Linear(edge_dim, out_dim, bias=False) if edge_dim else None
        )
        self.attn_src = nn.Parameter(torch.empty(1, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(1, out_dim))
        self.attn_edge = nn.Parameter(torch.empty(1, out_dim)) if edge_dim else None
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.activation = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.node_projection.weight)
        if self.edge_projection is not None:
            nn.init.xavier_normal_(self.edge_projection.weight)
            nn.init.xavier_normal_(self.attn_edge)
        nn.init.xavier_normal_(self.attn_src)
        nn.init.xavier_normal_(self.attn_dst)

    def forward(self, graph, node_features, edge_features=None, edge_weights=None):
        with graph.local_scope():
            projected = self.node_projection(node_features)
            if graph.num_edges() == 0:
                return self.activation(torch.zeros_like(projected) + self.bias)
            graph.ndata["projected"] = projected
            graph.ndata["score_src"] = (projected * self.attn_src).sum(-1, keepdim=True)
            graph.ndata["score_dst"] = (projected * self.attn_dst).sum(-1, keepdim=True)
            graph.apply_edges(fn.u_add_v("score_src", "score_dst", "score"))
            score = graph.edata["score"]
            if self.edge_projection is not None:
                if edge_features is None:
                    raise ValueError("edge features are required by this GAT layer")
                projected_edge = self.edge_projection(edge_features)
                score = score + (projected_edge * self.attn_edge).sum(-1, keepdim=True)
            attention = edge_softmax(graph, self.activation(score))
            if edge_weights is not None:
                attention = attention * edge_weights.reshape(-1, 1)
            graph.edata["attention"] = attention
            graph.update_all(
                fn.u_mul_e("projected", "attention", "message"),
                fn.sum("message", "updated"),
            )
            return self.activation(graph.ndata["updated"] + self.bias)


class IntraChannel(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, negative_slope=0.2, edge_dim=0):
        super().__init__()
        dimensions = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            GATLayer(dimensions[i], dimensions[i + 1], edge_dim, negative_slope)
            for i in range(num_layers)
        )

    def forward(self, graph, node_features, edge_features=None, edge_weights=None):
        hidden = node_features
        for layer in self.layers:
            hidden = layer(graph, hidden, edge_features, edge_weights)
        return hidden


class LigandAtomChannel(IntraChannel):
    def __init__(self, num_layers, in_dim, hidden_dim, negative_slope):
        super().__init__(num_layers, in_dim, hidden_dim, negative_slope)


class ProteinAtomChannel(LigandAtomChannel):
    pass


class LigandFragmentChannel(IntraChannel):
    def __init__(self, num_layers, in_dim, hidden_dim, negative_slope):
        super().__init__(num_layers, in_dim, hidden_dim, negative_slope)


class ProteinResidueChannel(IntraChannel):
    def __init__(self, num_layers, in_dim, hidden_dim, negative_slope):
        super().__init__(num_layers, in_dim, hidden_dim, negative_slope, edge_dim=1)


class _InterChannel(nn.Module):
    def __init__(self, num_layers, dimension, negative_slope=0.2):
        super().__init__()
        self.node_projections = nn.ModuleList(
            nn.Linear(dimension, dimension, bias=False) for _ in range(num_layers)
        )
        self.distance_projections = nn.ModuleList(
            nn.Linear(1, dimension, bias=False) for _ in range(num_layers)
        )
        self.attn_src = nn.ParameterList(
            nn.Parameter(torch.empty(1, dimension)) for _ in range(num_layers)
        )
        self.attn_dst = nn.ParameterList(
            nn.Parameter(torch.empty(1, dimension)) for _ in range(num_layers)
        )
        self.attn_edge = nn.ParameterList(
            nn.Parameter(torch.empty(1, dimension)) for _ in range(num_layers)
        )
        self.activation = nn.LeakyReLU(negative_slope)
        self.mechanics = InteractionForceMLP(atom_dim=dimension, hidden_dim=64)
        self.energy_fusion = nn.Linear(3, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for module in list(self.node_projections) + list(self.distance_projections):
            nn.init.xavier_normal_(module.weight)
        for parameter in (
            list(self.attn_src) + list(self.attn_dst) + list(self.attn_edge)
        ):
            nn.init.xavier_normal_(parameter)
        nn.init.constant_(self.energy_fusion.bias, 2.0)

    def _physical_weights(self, graph, hidden):
        src, dst = graph.edges()
        side = graph.ndata["side"]
        ligand_features = torch.where(
            (side[src] == 0).unsqueeze(-1), hidden[src], hidden[dst]
        )
        protein_features = torch.where(
            (side[src] == 1).unsqueeze(-1), hidden[src], hidden[dst]
        )
        vdw, electrostatic, hydrogen_bond = self.mechanics(
            ligand_features, protein_features, graph.edata["dist"]
        )
        return torch.sigmoid(
            self.energy_fusion(torch.cat((vdw, electrostatic, hydrogen_bond), dim=-1))
        )

    def forward(
        self,
        graph,
        h_ligand,
        h_protein,
        ligand_counts,
        protein_counts,
        edge_weights=None,
    ):
        hidden = pack_bipartite_features(
            h_ligand, h_protein, ligand_counts, protein_counts
        )
        if graph.num_edges() == 0:
            zeros = torch.zeros_like(hidden)
            return unpack_bipartite_features(zeros, ligand_counts, protein_counts)
        contact_gate = (
            self._physical_weights(graph, hidden)
            if edge_weights is None
            else edge_weights.reshape(-1, 1)
        )

        with graph.local_scope():
            for layer_id, projection in enumerate(self.node_projections):
                projected = projection(hidden)
                graph.ndata["projected"] = projected
                graph.ndata["score_src"] = (projected * self.attn_src[layer_id]).sum(
                    -1, keepdim=True
                )
                graph.ndata["score_dst"] = (projected * self.attn_dst[layer_id]).sum(
                    -1, keepdim=True
                )
                graph.apply_edges(fn.u_add_v("score_src", "score_dst", "score"))
                distance_feature = self.distance_projections[layer_id](
                    graph.edata["dist"]
                )
                score = graph.edata["score"] + (
                    distance_feature * self.attn_edge[layer_id]
                ).sum(-1, keepdim=True)
                attention = edge_softmax(graph, self.activation(score)) * contact_gate
                graph.edata["attention"] = attention
                graph.update_all(
                    fn.u_mul_e("projected", "attention", "message"),
                    fn.sum("message", "updated"),
                )
                hidden = self.activation(graph.ndata["updated"])

        return unpack_bipartite_features(hidden, ligand_counts, protein_counts)


class InterAtomChannel(_InterChannel):
    pass


class InterSubstructureChannel(_InterChannel):
    pass
