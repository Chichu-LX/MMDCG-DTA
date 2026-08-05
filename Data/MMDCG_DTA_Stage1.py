"""Stage-1 molecular-mechanics-informed MMDCG-DTA backbone."""

from __future__ import annotations

import dgl
import torch
import torch.nn as nn

from .channels import (
    InterAtomChannel,
    InterSubstructureChannel,
    LigandAtomChannel,
    LigandFragmentChannel,
    ProteinAtomChannel,
    ProteinResidueChannel,
    pack_bipartite_features,
)
from .hil import (
    AtomLevelInteractiveLigand,
    AtomLevelInteractiveProtein,
    SubstructureLevelInteractiveLigand,
    SubstructureLevelInteractiveProtein,
)
from .mechanics import IntraPhysicalGate


class MMDCGDTAModel_Stage1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_atom_hidden = int(config["embedding_dim"])
        self.d_sub_hidden = int(
            config.get("substructure_embedding_dim", self.d_atom_hidden)
        )
        self.use_checkpoint = bool(config.get("use_checkpoint", True))
        negative_slope = float(config["inter_negative_slope"])
        temperature = float(config.get("hil_temperature", 1.0))
        covalent_hidden_dim = int(config.get("covalent_hidden_dim", 32))
        noncovalent_hidden_dim = int(config.get("noncovalent_hidden_dim", 64))
        raw_atom_dim = int(config.get("raw_atom_dim", 5))

        self.ligand_atom_embedding = nn.Linear(raw_atom_dim, self.d_atom_hidden)
        self.protein_atom_embedding = nn.Linear(raw_atom_dim, self.d_atom_hidden)
        self.ligand_atom_intra_encoder = LigandAtomChannel(
            config["l_intra"],
            self.d_atom_hidden,
            self.d_atom_hidden,
            negative_slope,
        )
        self.protein_atom_intra_encoder = ProteinAtomChannel(
            config["l_intra"],
            self.d_atom_hidden,
            self.d_atom_hidden,
            negative_slope,
        )
        self.inter_atom_encoder = InterAtomChannel(
            config["l_inter"],
            self.d_atom_hidden,
            negative_slope,
            noncovalent_hidden_dim,
        )
        atom_hil = dict(
            num_steps=config["l_atom"],
            dimension=self.d_atom_hidden,
            temperature=temperature,
            negative_slope=negative_slope,
            use_checkpoint=self.use_checkpoint,
        )
        self.ligand_atom_interactive = AtomLevelInteractiveLigand(**atom_hil)
        self.protein_atom_interactive = AtomLevelInteractiveProtein(**atom_hil)

        ligand_sub_input = config.get("sub_x_dim", 5) + self.d_atom_hidden
        protein_sub_input = config.get("prot_res_dim", 1) + self.d_atom_hidden
        self.fragment_projection = nn.Linear(ligand_sub_input, self.d_sub_hidden)
        self.residue_projection = nn.Linear(protein_sub_input, self.d_sub_hidden)
        self.ligand_fragment_intra_encoder = LigandFragmentChannel(
            config["l_intra"], self.d_sub_hidden, self.d_sub_hidden, negative_slope
        )
        self.protein_residue_intra_encoder = ProteinResidueChannel(
            config["l_intra"], self.d_sub_hidden, self.d_sub_hidden, negative_slope
        )
        self.inter_substructure_encoder = InterSubstructureChannel(
            config["l_inter"],
            self.d_sub_hidden,
            negative_slope,
            noncovalent_hidden_dim,
        )
        sub_hil = dict(
            num_steps=config["l_sub"],
            dimension=self.d_sub_hidden,
            temperature=temperature,
            negative_slope=negative_slope,
            use_checkpoint=self.use_checkpoint,
        )
        self.ligand_substructure_interactive = SubstructureLevelInteractiveLigand(
            **sub_hil
        )
        self.protein_substructure_interactive = SubstructureLevelInteractiveProtein(
            **sub_hil
        )

        # Eq. (intra_physical_gate) uses one hierarchy-specific projection of
        # stretching, angular, and torsional channels at every GAT layer.
        self.atom_intra_physics = IntraPhysicalGate(
            self.d_atom_hidden, covalent_hidden_dim
        )
        self.substructure_intra_physics = IntraPhysicalGate(
            self.d_sub_hidden, covalent_hidden_dim
        )

        fusion_dimension = 4 * self.d_sub_hidden + 9
        readout_hidden_dim = int(config.get("readout_hidden_dim", 64))
        self.readout_gru = nn.GRU(
            input_size=fusion_dimension,
            hidden_size=readout_hidden_dim,
            batch_first=True,
        )
        self.affinity_regressor = nn.Linear(readout_hidden_dim, 1)

    @staticmethod
    def _batch_size(graph):
        return len(graph.batch_num_nodes())

    @staticmethod
    def _batch_group_ids(atom_graph, substructure_graph):
        device = atom_graph.device
        sub_counts = substructure_graph.batch_num_nodes().to(device)
        sub_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=device),
                torch.cumsum(sub_counts, dim=0)[:-1],
            )
        )
        atom_counts = atom_graph.batch_num_nodes().to(device)
        offsets_per_atom = torch.repeat_interleave(sub_offsets, atom_counts)
        if "group" not in atom_graph.ndata:
            raise ValueError(
                "graph cache has no exact atom-to-substructure mapping; rebuild it with "
                "`python -m Data.build_graph_dataset`"
            )
        local_groups = atom_graph.ndata["group"].long()
        if torch.any(local_groups < 0):
            raise ValueError("negative atom-to-substructure group ID")
        return local_groups + offsets_per_atom

    @staticmethod
    def _sample_group_ids(graph):
        counts = graph.batch_num_nodes().to(graph.device)
        return torch.repeat_interleave(
            torch.arange(len(counts), device=graph.device), counts
        )

    @staticmethod
    def _group_mean(values, group_ids, num_groups):
        totals = torch.zeros(
            num_groups, values.shape[-1], device=values.device, dtype=values.dtype
        )
        totals.index_add_(0, group_ids, values)
        counts = torch.zeros(num_groups, 1, device=values.device, dtype=values.dtype)
        counts.index_add_(
            0,
            group_ids,
            torch.ones(values.shape[0], 1, device=values.device, dtype=values.dtype),
        )
        return totals / counts.clamp_min(1.0)

    def _interaction_terms(
        self,
        encoder,
        graph,
        ligand_hidden,
        protein_hidden,
        ligand_counts,
        protein_counts,
    ):
        if graph.num_edges() == 0:
            zeros = torch.zeros(self._batch_size(graph), 1, device=graph.device)
            return (zeros, zeros.clone(), zeros.clone()), torch.empty(
                0, 1, device=graph.device
            )
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
        vdw, electrostatic, hydrogen_bond = encoder.mechanics(
            ligand_features, protein_features, graph.edata["dist"]
        )
        physical_weights = torch.sigmoid(
            encoder.energy_fusion(
                torch.cat((vdw, electrostatic, hydrogen_bond), dim=-1)
            )
        )
        with graph.local_scope():
            graph.edata["vdw"] = vdw
            graph.edata["electrostatic"] = electrostatic
            graph.edata["hydrogen_bond"] = hydrogen_bond
            # Every undirected candidate is represented by two directed edges.
            summaries = tuple(
                0.5 * dgl.readout_edges(graph, key, op="sum")
                for key in ("vdw", "electrostatic", "hydrogen_bond")
            )
        return summaries, physical_weights

    @staticmethod
    def _mean_pool(graph, hidden):
        with graph.local_scope():
            graph.ndata["readout"] = hidden
            return dgl.readout_nodes(graph, "readout", op="mean")

    def _reconstruct_contacts(self, hierarchy, graph, ligand_hidden, protein_hidden):
        """Stage-1 hook: physical gates are used instead of dynamic weights."""
        return None, {}

    def _forward_backbone(self, sample, dynamic_contacts=False):
        ligand_atom_graph = sample["ligand_atom_graph"]
        protein_atom_graph = sample["protein_atom_graph"]
        ligand_fragment_graph = sample["ligand_fragment_graph"]
        protein_residue_graph = sample["protein_residue_graph"]
        atom_graph = (
            sample["atom_candidate_graph"]
            if dynamic_contacts
            else sample["atom_interaction_graph"]
        )
        substructure_graph = sample["substructure_interaction_graph"]
        ligand_atom_counts = ligand_atom_graph.batch_num_nodes()
        protein_atom_counts = protein_atom_graph.batch_num_nodes()
        ligand_sub_counts = ligand_fragment_graph.batch_num_nodes()
        protein_sub_counts = protein_residue_graph.batch_num_nodes()

        ligand_atom_initial = self.ligand_atom_embedding(ligand_atom_graph.ndata["h"])
        protein_atom_initial = self.protein_atom_embedding(
            protein_atom_graph.ndata["h"]
        )
        ligand_atom_intra = self.ligand_atom_intra_encoder(
            ligand_atom_graph,
            ligand_atom_initial,
            self.atom_intra_physics,
        )
        protein_atom_intra = self.protein_atom_intra_encoder(
            protein_atom_graph,
            protein_atom_initial,
            self.atom_intra_physics,
        )
        atom_physics, atom_physical_weights = self._interaction_terms(
            self.inter_atom_encoder,
            atom_graph,
            ligand_atom_intra,
            protein_atom_intra,
            ligand_atom_counts,
            protein_atom_counts,
        )
        atom_weights, atom_auxiliary = (
            self._reconstruct_contacts(
                "atom", atom_graph, ligand_atom_intra, protein_atom_intra
            )
            if dynamic_contacts
            else (atom_physical_weights, {})
        )
        ligand_atom_inter, protein_atom_inter = self.inter_atom_encoder(
            atom_graph,
            ligand_atom_initial,
            protein_atom_initial,
            ligand_atom_counts,
            protein_atom_counts,
            edge_weights=atom_weights,
        )

        ligand_groups = self._batch_group_ids(ligand_atom_graph, ligand_fragment_graph)
        protein_groups = self._batch_group_ids(
            protein_atom_graph, protein_residue_graph
        )
        ligand_atom_inter, ligand_atom_intra = self.ligand_atom_interactive(
            ligand_atom_intra, ligand_atom_inter, ligand_groups
        )
        protein_atom_inter, protein_atom_intra = self.protein_atom_interactive(
            protein_atom_intra, protein_atom_inter, protein_groups
        )

        ligand_atom_mean = self._group_mean(
            ligand_atom_intra, ligand_groups, ligand_fragment_graph.num_nodes()
        )
        protein_atom_mean = self._group_mean(
            protein_atom_intra, protein_groups, protein_residue_graph.num_nodes()
        )
        ligand_sub_input = self.fragment_projection(
            torch.cat((ligand_fragment_graph.ndata["h"], ligand_atom_mean), dim=-1)
        )
        protein_sub_input = self.residue_projection(
            torch.cat((protein_residue_graph.ndata["h"], protein_atom_mean), dim=-1)
        )

        ligand_sub_intra = self.ligand_fragment_intra_encoder(
            ligand_fragment_graph,
            ligand_sub_input,
            self.substructure_intra_physics,
        )
        protein_sub_intra = self.protein_residue_intra_encoder(
            protein_residue_graph,
            protein_sub_input,
            self.substructure_intra_physics,
        )
        sub_weights, sub_auxiliary = (
            self._reconstruct_contacts(
                "substructure", substructure_graph, ligand_sub_intra, protein_sub_intra
            )
            if dynamic_contacts
            else (None, {})
        )
        if dynamic_contacts:
            sub_contact_gate = sub_weights
        else:
            _sub_physics, sub_contact_gate = self._interaction_terms(
                self.inter_substructure_encoder,
                substructure_graph,
                ligand_sub_intra,
                protein_sub_intra,
                ligand_sub_counts,
                protein_sub_counts,
            )
        ligand_sub_inter, protein_sub_inter = self.inter_substructure_encoder(
            substructure_graph,
            ligand_sub_input,
            protein_sub_input,
            ligand_sub_counts,
            protein_sub_counts,
            edge_weights=sub_contact_gate,
        )

        ligand_sub_groups = self._sample_group_ids(ligand_fragment_graph)
        protein_sub_groups = self._sample_group_ids(protein_residue_graph)
        ligand_sub_inter, ligand_sub_intra = self.ligand_substructure_interactive(
            ligand_sub_intra, ligand_sub_inter, ligand_sub_groups
        )
        protein_sub_inter, protein_sub_intra = self.protein_substructure_interactive(
            protein_sub_intra, protein_sub_inter, protein_sub_groups
        )

        (
            ligand_bond,
            ligand_angle,
            ligand_torsion,
        ) = self.atom_intra_physics.graph_summaries(
            ligand_atom_graph, ligand_atom_intra
        )
        (
            protein_bond,
            protein_angle,
            protein_torsion,
        ) = self.atom_intra_physics.graph_summaries(
            protein_atom_graph, protein_atom_intra
        )

        graph_summary = torch.cat(
            (
                self._mean_pool(ligand_fragment_graph, ligand_sub_intra),
                self._mean_pool(protein_residue_graph, protein_sub_intra),
                self._mean_pool(ligand_fragment_graph, ligand_sub_inter),
                self._mean_pool(protein_residue_graph, protein_sub_inter),
            ),
            dim=-1,
        )
        physics_summary = torch.cat(
            (
                ligand_bond,
                ligand_angle,
                ligand_torsion,
                protein_bond,
                protein_angle,
                protein_torsion,
                *atom_physics,
            ),
            dim=-1,
        )
        final_features = torch.cat((graph_summary, physics_summary), dim=-1).unsqueeze(
            1
        )
        fused, _ = self.readout_gru(final_features)
        prediction = self.affinity_regressor(fused.squeeze(1))
        return prediction, {"atom": atom_auxiliary, "substructure": sub_auxiliary}

    def forward(self, sample):
        prediction, _auxiliary = self._forward_backbone(sample, dynamic_contacts=False)
        return prediction
