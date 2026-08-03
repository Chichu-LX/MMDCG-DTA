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
from Data.training import collate_samples


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

    def test_two_scale_reconstruction(self):
        model = MMDCGDTAModel_Stage2(CONFIG)
        prediction, auxiliary = model(self.batch)
        self.assertEqual(tuple(prediction.shape), (2, 1))
        self.assertEqual(
            auxiliary["atom"]["logits"].shape[0],
            self.batch["atom_candidate_graph"].num_edges(),
        )
        self.assertEqual(
            auxiliary["substructure"]["logits"].shape[0],
            self.batch["substructure_interaction_graph"].num_edges(),
        )
        prediction.sum().backward()

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


if __name__ == "__main__":
    unittest.main()
