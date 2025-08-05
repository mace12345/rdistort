import unittest
import rdistort as rd
import os


class Testrdistort(unittest.TestCase):

    def test_ReadXYZFilesFromDirectory(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_dir = os.path.join(current_dir, "Test")
        molecule_set = rd.MoleculeSet([])
        molecule_set.ReadXYZFilesFromDirectory(test_dir)
        self.assertEqual(len(molecule_set.MoleculesList), 3)

    def test_CompareFeAndMnMolecules(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_dir = os.path.join(current_dir, "Test")
        molecule_set = rd.MoleculeSet([])
        molecule_set.ReadXYZFilesFromDirectory(test_dir)

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

        self.assertEqual(measurement.GetCurrent_rdistort_value(), 0.638)

        result = measurement.minimize_angles_with_basin_hopping(niter=10000)

        print(result)


if __name__ == "__main__":
    unittest.main()
