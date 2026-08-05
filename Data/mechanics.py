"""Neural molecular-mechanics surrogate channels used by MMDCG-DTA."""

from __future__ import annotations

import dgl
import torch
import torch.nn as nn


class BondEnergyMLP(nn.Module):
    """Approximate a scalar stretching term from an edge distance."""

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, distances):
        return self.mlp(distances)


class AngleDihedralEnergyMLP(nn.Module):
    """Infer node-wise angular and torsional surrogate channels."""

    def __init__(self, in_dim, hidden_dim=32):
        super().__init__()
        self.angle_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.torsion_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_states):
        return self.angle_mlp(node_states), self.torsion_mlp(node_states)


class IntraPhysicalGate(nn.Module):
    """Joint bond/angle/torsion gate from Eq. (intra_physical_gate)."""

    def __init__(self, state_dim, hidden_dim=32):
        super().__init__()
        self.bond_simulator = BondEnergyMLP(hidden_dim)
        self.angle_torsion_simulator = AngleDihedralEnergyMLP(state_dim, hidden_dim)
        self.channel_fusion = nn.Linear(3, 1)

    @staticmethod
    def _edge_distances(graph):
        source, destination = graph.edges()
        return torch.linalg.norm(
            graph.ndata["pos"][source] - graph.ndata["pos"][destination],
            dim=-1,
            keepdim=True,
        )

    def terms(self, graph, node_states):
        """Return edge gates and the node/edge surrogate terms for one layer."""
        angle_nodes, torsion_nodes = self.angle_torsion_simulator(node_states)
        if graph.num_edges() == 0:
            empty = node_states.new_empty((0, 1))
            return empty, {
                "bond_edges": empty,
                "angle_nodes": angle_nodes,
                "torsion_nodes": torsion_nodes,
                "angle_edges": empty,
                "torsion_edges": empty,
            }

        source, destination = graph.edges()
        bond_edges = self.bond_simulator(self._edge_distances(graph))
        angle_edges = 0.5 * (angle_nodes[source] + angle_nodes[destination])
        torsion_edges = 0.5 * (torsion_nodes[source] + torsion_nodes[destination])
        channels = torch.cat((bond_edges, angle_edges, torsion_edges), dim=-1)
        gates = torch.sigmoid(self.channel_fusion(channels))
        return gates, {
            "bond_edges": bond_edges,
            "angle_nodes": angle_nodes,
            "torsion_nodes": torsion_nodes,
            "angle_edges": angle_edges,
            "torsion_edges": torsion_edges,
        }

    def forward(self, graph, node_states):
        gates, _terms = self.terms(graph, node_states)
        return gates

    def graph_summaries(self, graph, node_states):
        """Return graph-level means used by the nine-channel affinity readout."""
        _gates, terms = self.terms(graph, node_states)
        batch_size = len(graph.batch_num_nodes())
        if graph.num_edges() == 0:
            bond = node_states.new_zeros((batch_size, 1))
        else:
            with graph.local_scope():
                graph.edata["bond_surrogate"] = terms["bond_edges"]
                bond = dgl.readout_edges(graph, "bond_surrogate", op="mean")
        with graph.local_scope():
            graph.ndata["angle_surrogate"] = terms["angle_nodes"]
            graph.ndata["torsion_surrogate"] = terms["torsion_nodes"]
            angle = dgl.readout_nodes(graph, "angle_surrogate", op="mean")
            torsion = dgl.readout_nodes(graph, "torsion_surrogate", op="mean")
        return bond, angle, torsion


class InteractionForceMLP(nn.Module):
    """Approximate van der Waals, electrostatic, and hydrogen-bond channels."""

    def __init__(self, atom_dim, hidden_dim=64):
        super().__init__()
        input_dim = atom_dim * 2 + 1

        def simulator():
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.vdw_simulator = simulator()
        self.electrostatic_simulator = simulator()
        self.hydrogen_bond_simulator = simulator()

    def forward(self, ligand_states, protein_states, distances):
        inputs = torch.cat((ligand_states, protein_states, distances), dim=-1)
        return (
            self.vdw_simulator(inputs),
            self.electrostatic_simulator(inputs),
            self.hydrogen_bond_simulator(inputs),
        )
