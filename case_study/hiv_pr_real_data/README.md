# HIV-1 Protease Complex Dataset

Generated: 2026-05-24 15:32:11

## Data Sources

- **RCSB PDB** (https://www.rcsb.org/): Structure files (.pdb format)
  - EC 3.4.23.16 (HIV-1 retropepsin), X-ray crystallography
  - Ligand SDF from RCSB Chemical Component Dictionary

- **PDBe** (https://www.ebi.ac.uk/pdbe/): Ligand monomer annotations

- **ChEMBL** (https://www.ebi.ac.uk/chembl/): Binding affinity data
  - Target: CHEMBL243 (HIV-1 protease)
  - Cross-referenced via InChIKey generated from PDB ligand SDF files

## Cross-Referencing Method

1. Downloaded SDF files for all unique PDB ligand 3-letter codes
2. Generated InChIKeys using RDKit
3. Searched ChEMBL molecule database by InChIKey
4. Filtered activities for target CHEMBL243 with Ki/Kd in nM
5. Matched back to PDB entries via ligand comp_id

## Dataset Statistics

- Total complexes with binding data: 50
- Resolution range: 1.06 - 2.50 Å
- Median resolution: 1.80 Å
- pKd range: 11.00 - 13.82
- Median pKd: 11.23

## Top 50 Complexes

| # | PDB ID | Resolution | Ligand | pKd | Type | ChEMBL ID |
|---|--------|-----------|--------|-----|------|-----------|
| 1 | 2FDD | 1.58 | (3R,3AS,6AR)-HEXAHYDROFURO[2,3-B]FURAN-3 | 13.82 | Ki | CHEMBL206031 |
| 2 | 2I0D | 1.95 | (5S)-3-(3-ACETYLPHENYL)-N-[(1S,2R)-1-BEN | 12.10 | Ki | CHEMBL384438 |
| 3 | 2O4S | 1.54 | N-{1-BENZYL-4-[2-(2,6-DIMETHYL-PHENOXY)- | 11.89 | Ki | CHEMBL729 |
| 4 | 2Q5K | 1.95 | N-{1-BENZYL-4-[2-(2,6-DIMETHYL-PHENOXY)- | 11.89 | Ki | CHEMBL729 |
| 5 | 2RKF | 1.8 | N-{1-BENZYL-4-[2-(2,6-DIMETHYL-PHENOXY)- | 11.89 | Ki | CHEMBL729 |
| 6 | 2RKG | 1.8 | N-{1-BENZYL-4-[2-(2,6-DIMETHYL-PHENOXY)- | 11.89 | Ki | CHEMBL729 |
| 7 | 2Z54 | 2.31 | N-{1-BENZYL-4-[2-(2,6-DIMETHYL-PHENOXY)- | 11.89 | Ki | CHEMBL729 |
| 8 | 4L1A | 1.9 | N-{1-BENZYL-4-[2-(2,6-DIMETHYL-PHENOXY)- | 11.89 | Ki | CHEMBL729 |
| 9 | 6DJ1 | 1.26 | N-{1-BENZYL-4-[2-(2,6-DIMETHYL-PHENOXY)- | 11.89 | Ki | CHEMBL729 |
| 10 | 6DJ2 | 1.36 | N-{1-BENZYL-4-[2-(2,6-DIMETHYL-PHENOXY)- | 11.89 | Ki | CHEMBL729 |
| 11 | 6PJB | 1.984 | N-{1-BENZYL-4-[2-(2,6-DIMETHYL-PHENOXY)- | 11.89 | Ki | CHEMBL729 |
| 12 | 9G35 | 1.5 | N-{1-BENZYL-4-[2-(2,6-DIMETHYL-PHENOXY)- | 11.89 | Ki | CHEMBL729 |
| 13 | 2I0A | 1.8 | (5S)-3-(4-ACETYLPHENYL)-N-[(1S,2R)-1-BEN | 11.40 | Ki | CHEMBL219748 |
| 14 | 8F0F | 1.29 | (3R,3aS,6aR)-hexahydrofuro[2,3-b]furan-3 | 11.36 | Ki | CHEMBL5287201 |
| 15 | 1PRO | 1.8 | (5R,6R)-2,4-BIS-(4-HYDROXY-3-METHOXYBENZ | 11.30 | Ki | CHEMBL443030 |
| 16 | 4U8W | 1.3 | (3R,3aS,6aS)-4,4-difluorohexahydrofuro[2 | 11.24 | Ki | CHEMBL3577575 |
| 17 | 4YHQ | 1.3 | (3R,3aS,6aS)-4,4-difluorohexahydrofuro[2 | 11.24 | Ki | CHEMBL3577575 |
| 18 | 6DJ7 | 1.31 | (3R,3aS,6aS)-4,4-difluorohexahydrofuro[2 | 11.24 | Ki | CHEMBL3577575 |
| 19 | 3OK9 | 1.27 | (3R,3aS,3bR,6aS,7aS)-octahydrodifuro[2,3 | 11.23 | Ki | CHEMBL1232930 |
| 20 | 4HDB | 1.49 | (3R,3aS,3bR,6aS,7aS)-octahydrodifuro[2,3 | 11.23 | Ki | CHEMBL1232930 |
| 21 | 4HDF | 1.29 | (3R,3aS,3bR,6aS,7aS)-octahydrodifuro[2,3 | 11.23 | Ki | CHEMBL1232930 |
| 22 | 4HDP | 1.22 | (3R,3aS,3bR,6aS,7aS)-octahydrodifuro[2,3 | 11.23 | Ki | CHEMBL1232930 |
| 23 | 4HE9 | 1.06 | (3R,3aS,3bR,6aS,7aS)-octahydrodifuro[2,3 | 11.23 | Ki | CHEMBL1232930 |
| 24 | 4HEG | 1.46 | (3R,3aS,3bR,6aS,7aS)-octahydrodifuro[2,3 | 11.23 | Ki | CHEMBL1232930 |
| 25 | 4J54 | 1.55 | (3R,3aS,3bR,6aS,7aS)-octahydrodifuro[2,3 | 11.23 | Ki | CHEMBL1232930 |
| 26 | 4RVI | 1.99 | (3R,3aS,3bR,6aS,7aS)-octahydrodifuro[2,3 | 11.23 | Ki | CHEMBL1232930 |
| 27 | 6DJ5 | 1.75 | (3R,3aS,3bR,6aS,7aS)-octahydrodifuro[2,3 | 11.23 | Ki | CHEMBL1232930 |
| 28 | 8DCH | 1.25 | (3R,3aS,3bR,6aS,7aS)-octahydrodifuro[2,3 | 11.23 | Ki | CHEMBL1232930 |
| 29 | 3GI5 | 1.8 | (5S)-3-(3-Acetylphenyl)-N-[(1S,2R)-3-[(1 | 11.22 | Ki | CHEMBL220078 |
| 30 | 3GI6 | 1.84 | (5S)-N-[(1S,2R)-2-Hydroxy-3-[[(4-methoxy | 11.22 | Ki | CHEMBL382400 |
| 31 | 1D4S | 2.5 | N-(3-{(1R)-1-[(6R)-4-HYDROXY-2-OXO-6-PHE | 11.10 | Ki | CHEMBL222559 |
| 32 | 1D4Y | 1.97 | N-(3-{(1R)-1-[(6R)-4-HYDROXY-2-OXO-6-PHE | 11.10 | Ki | CHEMBL222559 |
| 33 | 2O4L | 1.33 | N-(3-{(1R)-1-[(6R)-4-HYDROXY-2-OXO-6-PHE | 11.10 | Ki | CHEMBL222559 |
| 34 | 2O4N | 2.0 | N-(3-{(1R)-1-[(6R)-4-HYDROXY-2-OXO-6-PHE | 11.10 | Ki | CHEMBL222559 |
| 35 | 2O4P | 1.8 | N-(3-{(1R)-1-[(6R)-4-HYDROXY-2-OXO-6-PHE | 11.10 | Ki | CHEMBL222559 |
| 36 | 3SPK | 1.24 | N-(3-{(1R)-1-[(6R)-4-HYDROXY-2-OXO-6-PHE | 11.10 | Ki | CHEMBL222559 |
| 37 | 4NJU | 1.8 | N-(3-{(1R)-1-[(6R)-4-HYDROXY-2-OXO-6-PHE | 11.10 | Ki | CHEMBL222559 |
| 38 | 6DIF | 1.2 | N-(3-{(1R)-1-[(6R)-4-HYDROXY-2-OXO-6-PHE | 11.10 | Ki | CHEMBL222559 |
| 39 | 6DIL | 1.482 | N-(3-{(1R)-1-[(6R)-4-HYDROXY-2-OXO-6-PHE | 11.10 | Ki | CHEMBL222559 |
| 40 | 2I4W | 1.55 | DIETHYL ({4-[(2S,3R)-2-({[(3R,3AS,6AR)-H | 11.09 | Ki | CHEMBL1233845 |
| 41 | 2I4X | 1.55 | DIETHYL ({4-[(2S,3R)-2-({[(3R,3AS,6AR)-H | 11.09 | Ki | CHEMBL1233845 |
| 42 | 4M8X | 2.05 | DIETHYL ({4-[(2S,3R)-2-({[(3R,3AS,6AR)-H | 11.09 | Ki | CHEMBL1233845 |
| 43 | 4M8Y | 2.22 | DIETHYL ({4-[(2S,3R)-2-({[(3R,3AS,6AR)-H | 11.09 | Ki | CHEMBL1233845 |
| 44 | 7LE7 | 1.978 | DIETHYL ({4-[(2S,3R)-2-({[(3R,3AS,6AR)-H | 11.09 | Ki | CHEMBL1233845 |
| 45 | 7MAB | 1.879 | DIETHYL ({4-[(2S,3R)-2-({[(3R,3AS,6AR)-H | 11.09 | Ki | CHEMBL1233845 |
| 46 | 4I8Z | 1.75 | (3R,3aS,6aR)-hexahydrofuro[2,3-b]furan-3 | 11.05 | Ki | CHEMBL5271223 |
| 47 | 4NJS | 1.8 | (3R,3aS,6aR)-hexahydrofuro[2,3-b]furan-3 | 11.05 | Ki | CHEMBL5271223 |
| 48 | 1OHR | 2.1 | 2-[2-HYDROXY-3-(3-HYDROXY-2-METHYL-BENZO | 11.00 | Ki | CHEMBL584 |
| 49 | 2PYM | 1.9 | 2-[2-HYDROXY-3-(3-HYDROXY-2-METHYL-BENZO | 11.00 | Ki | CHEMBL584 |
| 50 | 2PYN | 1.85 | 2-[2-HYDROXY-3-(3-HYDROXY-2-METHYL-BENZO | 11.00 | Ki | CHEMBL584 |