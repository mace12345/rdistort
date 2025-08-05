import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.optimize import basinhopping
import os
import glob


class Atom:
    def __init__(
        self,
        Coordinates: np.ndarray,
        AtomicSymbol: str,
    ):
        self.Coordinates = Coordinates
        self.AtomicSymbol = AtomicSymbol


class Molecule:
    def __init__(
        self,
        Identifier: str,
        Atoms: list,
    ):
        self.Identifier = Identifier
        self.Atoms = Atoms or []

    def Measure_rdistort(
        self,
        ReferenceMolecule: "Molecule",
        ReferenceAtomMetalCenterIndex: int,
        ReferenceAtomLigandIndex: list,
        AtomMetalCenterIndex: int,
        AtomLigandIndex: list,
    ):
        pass


class MoleculeSet:
    def __init__(self, Molecules: list):
        self.MoleculesList = Molecules or []
        self.MoleculesDict = {m.Identifier: m for m in self.MoleculesList}

    def ReadXYZFilesFromDirectory(self, Directory: str):
        for filename in glob.glob(os.path.join(Directory, "*.xyz")):
            with open(filename, "r") as file:
                lines = file.readlines()
                identifier = os.path.splitext(os.path.basename(filename))[0]
                atoms = []
                for line in lines[2:]:
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    atomic_symbol = parts[0]
                    coordinates = np.array(
                        [float(parts[1]), float(parts[2]), float(parts[3])]
                    )
                    atoms.append(Atom(coordinates, atomic_symbol))
                self.MoleculesList.append(Molecule(identifier, atoms))
                self.MoleculesDict[identifier] = self.MoleculesList[-1]


class Measurement:
    def __init__(
        self,
        TestMolecule: Molecule,
        TestMoleculeAtomCenterIndex: int,
        TestMoleculeAtomLigandIndex: list,
        ReferenceMolecule: Molecule,
        ReferenceMoleculeAtomCenterIndex: int,
        ReferenceMoleculeAtomLigandIndex: list,
    ):
        self.TestMolecule = TestMolecule
        self.TestMoleculeAtomCenterIndex = TestMoleculeAtomCenterIndex
        self.TestMoleculeAtomLigandIndex = TestMoleculeAtomLigandIndex
        self.ReferenceMolecule = ReferenceMolecule
        self.ReferenceMoleculeAtomCenterIndex = ReferenceMoleculeAtomCenterIndex
        self.ReferenceMoleculeAtomLigandIndex = ReferenceMoleculeAtomLigandIndex
        self.rdistort_value = None
        self.optimal_xaxis_rotation = None
        self.optimal_yaxis_rotation = None
        self.optimal_zaxis_rotation = None

        self.test_vector_set = [
            self.TestMolecule.Atoms[i].Coordinates
            - self.TestMolecule.Atoms[self.TestMoleculeAtomCenterIndex].Coordinates
            for i in self.TestMoleculeAtomLigandIndex
        ]
        self.test_vector_set = [v / np.linalg.norm(v) for v in self.test_vector_set]
        self.reference_vector_set = [
            self.ReferenceMolecule.Atoms[i].Coordinates
            - self.ReferenceMolecule.Atoms[
                self.ReferenceMoleculeAtomCenterIndex
            ].Coordinates
            for i in self.ReferenceMoleculeAtomLigandIndex
        ]
        self.reference_vector_set = [
            v / np.linalg.norm(v) for v in self.reference_vector_set
        ]
        if len(self.test_vector_set) != len(self.reference_vector_set):
            raise ValueError(
                "Test and reference vector sets must have the same length.\nI.E. the ligand index lists must be the same length."
            )

    def Calculate_rdistort_value(self):
        # Calculate the magnitude matrix
        magnitude_matrix = np.zeros(
            (len(self.test_vector_set), len(self.reference_vector_set))
        )
        for i, test_vector in enumerate(self.test_vector_set):
            for j, reference_vector in enumerate(self.reference_vector_set):
                magnitude_matrix[i, j] = np.linalg.norm(test_vector + reference_vector)
        # Since linear_sum_assignment solves for MINIMUM cost,
        # negate the weights to convert to MAXIMUM assignment
        cost_matrix = -magnitude_matrix
        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # Total maximum weight
        max_total_weight = magnitude_matrix[row_ind, col_ind].sum()
        # Calculate the rdistort value
        self.rdistort_value = round(
            (len(self.test_vector_set) * 2) - max_total_weight, 3
        )
        return self.rdistort_value

    def GetCurrent_rdistort_value(self):
        if self.rdistort_value is None:
            # Get vector coordinates for the test and reference molecules
            self.Calculate_rdistort_value()
        return self.rdistort_value

    def RotationMatrix(self, rotation_axis: np.array, angle: float):
        """
        Create a rotation matrix for rotating around a given axis by a specified angle.
        """
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        ux, uy, uz = rotation_axis
        return np.array(
            [
                [
                    cos_angle + ux**2 * (1 - cos_angle),
                    ux * uy * (1 - cos_angle) - uz * sin_angle,
                    ux * uz * (1 - cos_angle) + uy * sin_angle,
                ],
                [
                    uy * ux * (1 - cos_angle) + uz * sin_angle,
                    cos_angle + uy**2 * (1 - cos_angle),
                    uy * uz * (1 - cos_angle) - ux * sin_angle,
                ],
                [
                    uz * ux * (1 - cos_angle) - uy * sin_angle,
                    uz * uy * (1 - cos_angle) + ux * sin_angle,
                    cos_angle + uz**2 * (1 - cos_angle),
                ],
            ]
        )

    def Calculate_rdistort_value_NewTest_vector_set(
        self,
        new_test_vector_set: list,
    ):
        # Calculate the magnitude matrix
        magnitude_matrix = np.zeros(
            (len(new_test_vector_set), len(self.reference_vector_set))
        )
        for i, test_vector in enumerate(new_test_vector_set):
            for j, reference_vector in enumerate(self.reference_vector_set):
                magnitude_matrix[i, j] = np.linalg.norm(test_vector + reference_vector)
        # Since linear_sum_assignment solves for MINIMUM cost,
        # negate the weights to convert to MAXIMUM assignment
        cost_matrix = -magnitude_matrix
        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # Total maximum weight
        max_total_weight = magnitude_matrix[row_ind, col_ind].sum()
        # Calculate the rdistort value
        self.rdistort_value = round(
            (len(self.test_vector_set) * 2) - max_total_weight, 3
        )
        return self.rdistort_value

    def Rotate_and_Calculate_rdistort_value(
        self,
        theta_degs: float,
        phi_degs: float,
        psi_degs: float,
    ):
        theta = np.deg2rad(theta_degs)
        phi = np.deg2rad(phi_degs)
        psi = np.deg2rad(psi_degs)
        x_rotation_axis = self.RotationMatrix(np.array([1, 0, 0]), theta)
        y_rotation_axis = self.RotationMatrix(np.array([0, 1, 0]), phi)
        z_rotation_axis = self.RotationMatrix(np.array([0, 0, 1]), psi)
        new_test_vector_set = [
            np.dot(x_rotation_axis, np.dot(y_rotation_axis, np.dot(z_rotation_axis, v)))
            for v in self.test_vector_set
        ]
        return self.Calculate_rdistort_value_NewTest_vector_set(new_test_vector_set)

    def minimize_angles_with_basin_hopping(
        self,
        niter: int = 500,
        stepsize: float = 100,
        T: float = 10,
        disp: bool = False,
    ):
        # Wrapper for the objective function
        def objective(x):
            theta, phi, psi = x
            return self.Rotate_and_Calculate_rdistort_value(theta, phi, psi)

        # Initial guess (in degrees)
        x0 = [0.0, 0.0, 0.0]

        # Optional: bounds or constraints (basinhopping supports them via 'minimizer_kwargs')
        bounds = [(0, 360), (0, 360), (0, 360)]
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}

        # Run basin hopping
        result = basinhopping(
            objective,
            x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=niter,
            stepsize=stepsize,
            T=T,
            disp=disp,
        )
        # Store the optimal angles and rdistort value
        self.optimal_xaxis_rotation = result.x[0]
        self.optimal_yaxis_rotation = result.x[1]
        self.optimal_zaxis_rotation = result.x[2]
        self.rdistort_value = result.fun
