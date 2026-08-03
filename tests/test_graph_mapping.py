import unittest

from rdkit import Chem
from rdkit.Chem import AllChem

from Data.graphs import (
    build_atom_interaction_graph,
    build_ligand_atom_graph,
    build_ligand_fragment_graph,
    build_protein_atom_graph,
    build_protein_residue_graph,
)


PROTEIN_PDB = """\
ATOM      1  N   ALA A   1       0.000   3.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.450   3.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.100   4.300   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       1.500   5.350   0.000  1.00 20.00           O
ATOM      5  N   GLY A   2       3.350   4.250   0.000  1.00 20.00           N
ATOM      6  CA  GLY A   2       4.100   5.450   0.000  1.00 20.00           C
ATOM      7  C   GLY A   2       5.550   5.100   0.000  1.00 20.00           C
ATOM      8  O   GLY A   2       6.350   6.000   0.000  1.00 20.00           O
TER
END
"""


class GraphMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        molecule = Chem.AddHs(Chem.MolFromSmiles("CCOC(=O)NCC"))
        AllChem.EmbedMolecule(molecule, randomSeed=42)
        cls.sdf = Chem.MolToMolBlock(molecule)

    def test_brics_memberships_cover_ligand_atoms(self):
        atoms = build_ligand_atom_graph(self.sdf)
        fragments = build_ligand_fragment_graph(self.sdf)
        groups = atoms.ndata["group"]
        self.assertEqual(groups.shape[0], atoms.num_nodes())
        self.assertEqual(int(groups.min()), 0)
        self.assertEqual(int(groups.max()) + 1, fragments.num_nodes())
        self.assertEqual(len(groups.unique()), fragments.num_nodes())

    def test_pdb_memberships_match_residue_nodes(self):
        atoms = build_protein_atom_graph(PROTEIN_PDB)
        residues = build_protein_residue_graph(PROTEIN_PDB, cutoff=8.0)
        groups = atoms.ndata["group"]
        self.assertEqual(atoms.num_nodes(), 8)
        self.assertEqual(residues.num_nodes(), 2)
        self.assertEqual(set(groups.tolist()), {0, 1})

    def test_candidate_graph_contains_initial_contacts(self):
        ligand = build_ligand_atom_graph(self.sdf)
        protein = build_protein_atom_graph(PROTEIN_PDB)
        initial = build_atom_interaction_graph(ligand, protein, 4.0)
        candidate = build_atom_interaction_graph(ligand, protein, 8.0)
        self.assertGreaterEqual(candidate.num_edges(), initial.num_edges())


if __name__ == "__main__":
    unittest.main()
