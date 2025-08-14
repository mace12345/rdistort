import unittest
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import rdistort.rdistort as rd


class test_rdistort(unittest.TestCase):

    def test_ReadXYZFilesFromDirectory(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        molecule_set = rd.MoleculeSet([])
        molecule_set.ReadXYZFilesFromDirectory(current_dir)
        self.assertEqual(len(molecule_set.MoleculesList), 2)

    def test_CompareFeAndMnMolecules(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        molecule_set = rd.MoleculeSet([])
        molecule_set.ReadXYZFilesFromDirectory(current_dir)

        fe_molecule = molecule_set.MoleculesDict["FeComplex"]
        mn_molecule = molecule_set.MoleculesDict["MnComplex"]

        measurement = rd.Measurement(
            TestMolecule=fe_molecule,
            TestMoleculeAtomCenterIndex=0,
            TestMoleculeAtomLigandIndex=[1, 3, 6, 9, 12, 25],
            ReferenceMolecule=mn_molecule,
            ReferenceMoleculeAtomCenterIndex=12,
            ReferenceMoleculeAtomLigandIndex=[0, 1, 9, 10, 13, 61],
        )

        self.assertEqual(measurement.GetCurrent_rdistort_value(), 0.63787)

        # measurement.Minimize_rdistort_BasisHopping()
        # self.assertLess(measurement.rdistort_value, 0.1)

        # measurement.Minimize_rdistort_KabschAlignment()
        # measurement.Minimize_rdistort_BasisHopping()
        # self.assertLess(measurement.rdistort_value, 0.05)

        # measurement.Minimize_rdistort_BruteForce()
        # self.assertLess(measurement.rdistort_value, 0.02)

        # measurement.Minimize_rdistort_EfficientBruteForce(
        #    grid_size_first_stage=10,
        #    grid_size_second_stage=1,
        # )
        # sself.assertLess(measurement.rdistort_value, 0.019)

    """def test_GetMass_rdistortValues(self):
        location = "C:/Users/samue/OneDrive - University of Leeds/Samuel Mace PhD Research Project/SpinStateV2/ResultsAnalysis/XYZ_FILES"

        methods = [
            ["PBE0", "", "def2-TZVP"],
            ["TPSSh", "", "def2-TZVP"],
            ["B3LYP", "", "def2-TZVP"],
            ["M06L", "", "def2-TZVP"],
        ]

        metals_dict = {
            "CrII": [1, 3, 5],
            "MnII": [2, 4, 6],
            "FeIII": [2, 4, 6],
            "FeII": [1, 3, 5],
            "CoII": [2, 4],
        }

        error_count = 0
        for metal in metals_dict:
            crystal_set = rd.MoleculeSet([])
            crystal_set.ReadXYZFilesFromDirectory(
                f"{location}/{metal}Lig_XRD-Structures"
            )
            indices_df = pd.read_csv(
                f"{location}/{metal}Lig_XRD-Structures_Metal_Ligand_Index.csv"
            )
            indices_df.rename(columns={"Unnamed: 0": "Identifier"}, inplace=True)
            indices_df.set_index("Identifier", inplace=True)
            for mult in metals_dict[metal]:
                for method in methods:
                    rd_df = pd.DataFrame()
                    DFT = method[0]
                    DFT_set = rd.MoleculeSet([])
                    DFT_set.ReadXYZFilesFromDirectory(
                        f"{location}/{metal}Lig-S{mult}_{DFT}-def2-TZVP-Opt-Freq-DEFGRID3_ORCA6Output"
                    )
                    for idx, row in indices_df.iterrows():
                        crystal_mol = crystal_set.MoleculesDict[idx]
                        dft_mol = DFT_set.MoleculesDict[idx]
                        metal_center_idx = int(row["Metal Atom"])
                        ligand_idx_list = []
                        row.dropna(inplace=True)
                        for i in range(0, row.size - 1):
                            ligand_idx = int(row[f"{i}"])
                            ligand_idx_list.append(ligand_idx)
                        try:
                            measurment = rd.Measurement(
                                TestMolecule=dft_mol,
                                TestMoleculeAtomCenterIndex=metal_center_idx,
                                TestMoleculeAtomLigandIndex=ligand_idx_list,
                                ReferenceMolecule=crystal_mol,
                                ReferenceMoleculeAtomCenterIndex=metal_center_idx,
                                ReferenceMoleculeAtomLigandIndex=ligand_idx_list,
                            )
                            measurment.Minimize_rdistort_NoMatchingVectorsKabschAlignment()
                            rd_df.loc[idx, "r-distort value"] = (
                                measurment.rdistort_value
                            )
                            rd_df.loc[idx, "Coordination Number"] = len(ligand_idx_list)
                        except IndexError:
                            error_count += 1
                    rd_df.to_csv(
                        f"{location}/{metal}Lig-S{mult}_{DFT}-def2-TZVP_rdistort_value.csv"
                    )

        print(error_count)"""


if __name__ == "__main__":
    unittest.main()
