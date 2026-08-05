import unittest

import dgl
import torch

from Data.MMDCG_DTA_Stage1 import MMDCGDTAModel_Stage1
from Data.MMDCG_DTA_Stage2 import MMDCGDTAModel_Stage2
from Data.MMDCG_DTA_Stage3 import MMDCGDTAModel_Stage3
from Data.graphs import (
    build_atom_interaction_graph,
    build_substructure_interaction_graph,
)
from Data.hil import AtomLevelInteractiveLigand
from Data.training import _combined_edge_loss, _set_stage2_trainable, collate_samples


CONFIG = {
    "embedding_dim": 64,
    "substructure_embedding_dim": 64,
    "raw_atom_dim": 5,
    "sub_x_dim": 5,
    "prot_res_dim": 1,
    "l_intra": 2,
    "l_inter": 2,
    "l_atom": 2,
    "l_sub": 2,
    "inter_negative_slope": 0.2,
    "hil_temperature": 1.0,
    "use_checkpoint": False,
    "covalent_hidden_dim": 32,
    "noncovalent_hidden_dim": 64,
    "edge_reconstructor_hidden_dim": 64,
    "readout_hidden_dim": 64,
}


def intra_graph(features, positions, groups=None, residue=False):
    num_nodes = len(positions)
    src = []
    dst = []
    for node in range(num_nodes - 1):
        src.extend((node, node + 1))
        dst.extend((node + 1, node))
    graph = dgl.graph((src, dst), num_nodes=num_nodes)
    graph.ndata["h"] = torch.tensor(features, dtype=torch.float32)
    graph.ndata["pos"] = torch.tensor(positions, dtype=torch.float32)
    if groups is not None:
        graph.ndata["group"] = torch.tensor(groups, dtype=torch.long)
    if residue:
        graph.edata["dist"] = torch.linalg.norm(
            graph.ndata["pos"][torch.tensor(src)]
            - graph.ndata["pos"][torch.tensor(dst)],
            dim=-1,
            keepdim=True,
        )
    return graph


def sample(offset, ligand_groups, protein_groups, num_fragments, num_residues):
    ligand_positions = [[offset + i * 0.8, 0.0, 0.0] for i in range(len(ligand_groups))]
    protein_positions = [
        [offset + i * 0.8, 2.8, 0.0] for i in range(len(protein_groups))
    ]
    ligand_atom = intra_graph(
        [[6, 0, 0, 2, 0.0]] * len(ligand_groups), ligand_positions, ligand_groups
    )
    protein_atom = intra_graph(
        [[6, 0, 0, 2, 0.0]] * len(protein_groups), protein_positions, protein_groups
    )
    fragment_positions = []
    for group_id in range(num_fragments):
        member_positions = [
            ligand_positions[i]
            for i, value in enumerate(ligand_groups)
            if value == group_id
        ]
        fragment_positions.append(torch.tensor(member_positions).mean(0).tolist())
    residue_positions = []
    for group_id in range(num_residues):
        member_positions = [
            protein_positions[i]
            for i, value in enumerate(protein_groups)
            if value == group_id
        ]
        residue_positions.append(torch.tensor(member_positions).mean(0).tolist())
    ligand_fragment = intra_graph(
        [[30, 1, 10, 0, 0]] * num_fragments, fragment_positions
    )
    protein_residue = intra_graph(
        [[12]] * num_residues, residue_positions, residue=True
    )
    atom_initial = build_atom_interaction_graph(ligand_atom, protein_atom, 4.0)
    atom_candidate = build_atom_interaction_graph(ligand_atom, protein_atom, 8.0)
    substructure = build_substructure_interaction_graph(
        ligand_fragment, protein_residue, 8.0
    )
    return {
        "compound_id": f"sample-{offset}",
        "ligand_atom_graph": ligand_atom,
        "protein_atom_graph": protein_atom,
        "atom_interaction_graph": atom_initial,
        "atom_candidate_graph": atom_candidate,
        "ligand_fragment_graph": ligand_fragment,
        "protein_residue_graph": protein_residue,
        "substructure_interaction_graph": substructure,
        "label": 7.0 + offset,
    }


class ModelSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = collate_samples(
            [
                sample(0.0, [0, 0, 1], [0, 0, 1, 1], 2, 2),
                sample(10.0, [0, 0], [0, 1, 1], 1, 2),
            ]
        )

    def test_stage1_batch_forward_and_backward(self):
        model = MMDCGDTAModel_Stage1(CONFIG)
        prediction = model(self.batch)
        self.assertEqual(tuple(prediction.shape), (2, 1))
        torch.nn.functional.mse_loss(prediction, self.batch["label"]).backward()
        self.assertEqual(model.readout_gru.hidden_size, 64)
        for hierarchy_gate in (
            model.atom_intra_physics,
            model.substructure_intra_physics,
        ):
            self.assertIsNotNone(hierarchy_gate.bond_simulator.mlp[0].weight.grad)
            self.assertIsNotNone(
                hierarchy_gate.angle_torsion_simulator.angle_mlp[0].weight.grad
            )
            self.assertIsNotNone(
                hierarchy_gate.angle_torsion_simulator.torsion_mlp[0].weight.grad
            )
            self.assertIsNotNone(hierarchy_gate.channel_fusion.weight.grad)
        for inter_encoder in (
            model.inter_atom_encoder,
            model.inter_substructure_encoder,
        ):
            self.assertIsNotNone(inter_encoder.mechanics.vdw_simulator[0].weight.grad)
            self.assertIsNotNone(
                inter_encoder.mechanics.electrostatic_simulator[0].weight.grad
            )
            self.assertIsNotNone(
                inter_encoder.mechanics.hydrogen_bond_simulator[0].weight.grad
            )
            self.assertIsNotNone(inter_encoder.energy_fusion.weight.grad)

    def test_two_scale_reconstruction(self):
        model = MMDCGDTAModel_Stage2(CONFIG)
        prediction, auxiliary = model(self.batch)
        self.assertEqual(tuple(prediction.shape), (2, 1))
        self.assertEqual(
            auxiliary["atom"]["logits"].shape[0],
            self.batch["atom_candidate_graph"].num_edges() // 2,
        )
        self.assertEqual(
            auxiliary["substructure"]["logits"].shape[0],
            self.batch["substructure_interaction_graph"].num_edges() // 2,
        )
        self.assertEqual(
            auxiliary["atom"]["stats"]["total"],
            self.batch["atom_candidate_graph"].num_edges() // 2,
        )
        atom_parameter_ids = {
            id(parameter) for parameter in model.atom_edge_reconstructor.parameters()
        }
        substructure_parameter_ids = {
            id(parameter)
            for parameter in model.substructure_edge_reconstructor.parameters()
        }
        self.assertTrue(atom_parameter_ids.isdisjoint(substructure_parameter_ids))
        prediction.sum().backward()

    def test_stage1_checkpoint_has_only_reconstructor_keys_missing_in_stage2(self):
        stage1 = MMDCGDTAModel_Stage1(CONFIG)
        stage2 = MMDCGDTAModel_Stage2(CONFIG)
        incompatible = stage2.load_state_dict(stage1.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all("edge_reconstructor" in key for key in incompatible.missing_keys)
        )

    def test_stage2_inner_and_outer_parameter_isolation(self):
        model = MMDCGDTAModel_Stage2(CONFIG)
        reconstructor_parameters = [
            parameter
            for reconstructor in model.reconstructors
            for parameter in reconstructor.parameters()
        ]
        reconstructor_ids = {id(parameter) for parameter in reconstructor_parameters}
        backbone_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in reconstructor_ids
        ]

        _set_stage2_trainable(model, reconstructors=True)
        _prediction, auxiliary = model(self.batch)
        edge_loss, _classes = _combined_edge_loss(
            auxiliary, self.batch, torch.nn.CrossEntropyLoss()
        )
        edge_loss.backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in reconstructor_parameters)
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in backbone_parameters)
        )

        model.zero_grad(set_to_none=True)
        _set_stage2_trainable(model, reconstructors=False)
        prediction, _auxiliary = model(self.batch)
        torch.nn.functional.mse_loss(prediction, self.batch["label"]).backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in reconstructor_parameters)
        )
        self.assertTrue(
            any(parameter.grad is not None for parameter in backbone_parameters)
        )

    def test_cross_hierarchy_paths_do_not_share_parameters(self):
        model = MMDCGDTAModel_Stage1(CONFIG)
        modules = (
            model.ligand_atom_interactive,
            model.protein_atom_interactive,
            model.ligand_substructure_interactive,
            model.protein_substructure_interactive,
        )
        parameter_sets = [
            {id(parameter) for parameter in module.parameters()} for module in modules
        ]
        for left in range(len(parameter_sets)):
            for right in range(left + 1, len(parameter_sets)):
                self.assertTrue(parameter_sets[left].isdisjoint(parameter_sets[right]))

    def test_cross_hierarchy_broadcast_is_group_local(self):
        torch.manual_seed(7)
        module = AtomLevelInteractiveLigand(
            num_steps=2,
            dimension=8,
            temperature=1.0,
            negative_slope=0.2,
            use_checkpoint=False,
        )
        intra = torch.randn(4, 8)
        inter = torch.randn(4, 8)
        groups = torch.tensor([0, 0, 1, 1])
        reference = module(intra, inter, groups)
        changed_intra = intra.clone()
        changed_intra[:2] += 100.0
        changed = module(changed_intra, inter, groups)
        self.assertTrue(torch.allclose(reference[0][2:], changed[0][2:]))
        self.assertTrue(torch.allclose(reference[1][2:], changed[1][2:]))

    def test_stage3_freezes_both_reconstructors(self):
        model = MMDCGDTAModel_Stage3(CONFIG)
        model.freeze_reconstructors()
        self.assertTrue(
            all(
                not parameter.requires_grad
                for reconstructor in model.reconstructors
                for parameter in reconstructor.parameters()
            )
        )
        prediction, _auxiliary = model(self.batch)
        prediction.sum().backward()
        self.assertIsNotNone(model.ligand_atom_embedding.weight.grad)
        self.assertTrue(
            all(
                parameter.grad is None
                for reconstructor in model.reconstructors
                for parameter in reconstructor.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
