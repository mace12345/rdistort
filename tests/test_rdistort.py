import unittest
import sys
import os

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

        measurement.Minimize_rdistort_BasisHopping()
        self.assertLess(measurement.rdistort_value, 0.1)

        measurement.Minimize_rdistort_KabschAlignment()
        measurement.Minimize_rdistort_BasisHopping()
        self.assertLess(measurement.rdistort_value, 0.05)

        measurement.Minimize_rdistort_BruteForce()
        self.assertLess(measurement.rdistort_value, 0.02)

        measurement.Minimize_rdistort_EfficientBruteForce(
            grid_size_first_stage=10,
            grid_size_second_stage=1,
        )
        self.assertLess(measurement.rdistort_value, 0.019)


if __name__ == "__main__":
    unittest.main()
