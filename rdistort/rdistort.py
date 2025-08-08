import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.optimize import basinhopping
import os
import glob
import itertools


class Atom:
    """
    A class representing an atom with its 3D coordinates and atomic symbol.

    Parameters
    ----------
    Coordinates : np.ndarray
        A 3-element NumPy array representing the (x, y, z) coordinates of the atom in 3D space.
    AtomicSymbol : str
        A string representing the atomic symbol (e.g., 'H', 'C', 'O') of the atom.

    Attributes
    ----------
    Coordinates : np.ndarray
        The spatial coordinates of the atom.
    AtomicSymbol : str
        The atomic symbol identifying the element type.
    """

    def __init__(
        self,
        Coordinates: np.ndarray,
        AtomicSymbol: str,
    ):
        self.Coordinates = Coordinates
        self.AtomicSymbol = AtomicSymbol


class Molecule:
    """
    A class representing a molecule composed of atoms.

    Parameters
    ----------
    Identifier : str
        A unique identifier or name for the molecule.
    Atoms : list of Atom
        A list of `Atom` instances that make up the molecule. If None or empty, an empty list is used.

    Attributes
    ----------
    Identifier : str
        The name or identifier of the molecule.
    Atoms : list of Atom
        The list of atoms in the molecule.
    """

    def __init__(
        self,
        Identifier: str,
        Atoms: list,
    ):
        self.Identifier = Identifier
        self.Atoms = Atoms or []


class MoleculeSet:
    """
    A class to represent a collection of molecules and facilitate loading them from XYZ files.

    Parameters
    ----------
    Molecules : list of Molecule
        A list of `Molecule` instances to initialize the set with. If None or empty, initializes an empty set.

    Attributes
    ----------
    MoleculesList : list of Molecule
        List of all `Molecule` instances in the set.
    MoleculesDict : dict of str -> Molecule
        Dictionary mapping molecule identifiers to their corresponding `Molecule` objects.
    """

    def __init__(self, Molecules: list):
        self.MoleculesList = Molecules or []
        self.MoleculesDict = {m.Identifier: m for m in self.MoleculesList}

    def ReadXYZFilesFromDirectory(self, Directory: str):
        """
        Reads all `.xyz` files in the specified directory and adds the corresponding molecules
        to the molecule set.

        Each `.xyz` file is expected to follow the standard format:
        Line 1: Number of atoms (ignored)
        Line 2: Comment (ignored)
        Remaining lines: Atomic symbol and x, y, z coordinates

        Parameters
        ----------
        Directory : str
            Path to the directory containing `.xyz` files.

        Notes
        -----
        - Files are parsed based on filename to use as molecule identifier (excluding extension).
        - Files with malformed atom lines (fewer than 4 components) are skipped.
        - Molecules are added both to `MoleculesList` and `MoleculesDict`.
        """
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
    """
    A class for comparing the geometric relationship between a test molecule and a reference molecule
    based on selected atoms.

    This class computes normalized directional vectors from a central atom to a set of ligand atoms
    in both the test and reference molecules. It checks that the number of vectors matches and prepares
    the system for alignment or distortion analysis.

    Parameters
    ----------
    TestMolecule : Molecule
        The molecule whose geometry is being tested.
    TestMoleculeAtomCenterIndex : int
        Index of the central atom in the test molecule.
    TestMoleculeAtomLigandIndex : list of int
        Indices of ligand atoms in the test molecule (atoms connected to the central atom).
    ReferenceMolecule : Molecule
        The molecule used as the geometric reference.
    ReferenceMoleculeAtomCenterIndex : int
        Index of the central atom in the reference molecule.
    ReferenceMoleculeAtomLigandIndex : list of int
        Indices of ligand atoms in the reference molecule.

    Attributes
    ----------
    TestMolecule : Molecule
        The test molecule object.
    TestMoleculeAtomCenterIndex : int
        Index of the central atom in the test molecule.
    TestMoleculeAtomLigandIndex : list of int
        List of indices for ligand atoms in the test molecule.
    ReferenceMolecule : Molecule
        The reference molecule object.
    ReferenceMoleculeAtomCenterIndex : int
        Index of the central atom in the reference molecule.
    ReferenceMoleculeAtomLigandIndex : list of int
        List of indices for ligand atoms in the reference molecule.
    rdistort_value : float or None
        Placeholder for computed distortion value (if calculated later).
    optimal_xaxis_rotation : float or None
        Placeholder for optimal rotation about the X-axis.
    optimal_yaxis_rotation : float or None
        Placeholder for optimal rotation about the Y-axis.
    optimal_zaxis_rotation : float or None
        Placeholder for optimal rotation about the Z-axis.
    test_vector_set : list of np.ndarray
        Normalized vectors from the central atom to ligand atoms in the test molecule.
    reference_vector_set : list of np.ndarray
        Normalized vectors from the central atom to ligand atoms in the reference molecule.

    Raises
    ------
    ValueError
        If the number of ligand atoms in the test and reference molecules do not match.
    """

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
        """
        Calculates the rdistort value, a geometric distortion metric between
        the test and reference vector sets.

        The rdistort value is computed by:
        1. Creating a matrix of vector sums between all test and reference vectors.
        2. Using the Hungarian algorithm to find the optimal matching (maximum alignment).
        3. Subtracting the optimal alignment score from the maximum possible score.

        Returns
        -------
        float
            The computed rdistort value, rounded to 3 decimal places. Lower values
            indicate better geometric similarity between the test and reference vector sets.

        Notes
        -----
        - This method modifies the internal `rdistort_value` attribute.
        - The method assumes both vector sets are already normalized and of equal length.
        """
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
            (len(self.test_vector_set) * 2) - max_total_weight, 5
        )
        return self.rdistort_value

    def GetCurrent_rdistort_value(self):
        """
        Returns the current rdistort value, computing it if necessary.

        Returns
        -------
        float
            The cached or newly computed rdistort value.

        Notes
        -----
        If `rdistort_value` has not yet been calculated, this method will call
        `Calculate_rdistort_value()` to compute and cache it.
        """
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
        """
        Calculates the rdistort value using a custom test vector set instead of the default one.

        This allows evaluation of how a modified (e.g., rotated) version of the test vector set
        compares to the reference vector set in terms of geometric distortion.

        Parameters
        ----------
        new_test_vector_set : list of np.ndarray
            A list of normalized vectors representing the modified test geometry.

        Returns
        -------
        float
            The computed rdistort value based on the new test vector set.

        Notes
        -----
        - Updates the internal `rdistort_value` attribute.
        - Uses the Hungarian algorithm to find optimal matching between vectors.
        """
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
            (len(self.test_vector_set) * 2) - max_total_weight, 5
        )
        return self.rdistort_value

    def Rotate_and_Calculate_rdistort_value(
        self,
        theta_degs: float,
        phi_degs: float,
        psi_degs: float,
    ):
        """
        Applies Euler angle rotations to the test vector set and calculates the resulting rdistort value.

        The rotation is composed of three sequential axis rotations:
        Z-axis → Y-axis → X-axis.

        Parameters
        ----------
        theta_degs : float
            Rotation angle (in degrees) about the X-axis.
        phi_degs : float
            Rotation angle (in degrees) about the Y-axis.
        psi_degs : float
            Rotation angle (in degrees) about the Z-axis.

        Returns
        -------
        float
            The rdistort value after applying the given rotations to the test vector set.

        Notes
        -----
        - Rotation matrices are applied in the order Z → Y → X.
        - Internally calls `Calculate_rdistort_value_NewTest_vector_set`.
        """
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

    def Minimize_rdistort_BasisHopping(
        self,
        niter: int = 500,
        stepsize: float = 100,
        T: float = 10,
        disp: bool = False,
        x0: list = [0.0, 0.0, 0.0],
        bounds: list = [(0, 360), (0, 360), (0, 360)],
    ):
        """
        Optimizes rotation angles (Euler angles) to minimize the rdistort value using the basin-hopping algorithm.

        This method searches over the space of possible 3D rotations to find the best alignment of
        the test vector set to the reference vector set that yields the minimum rdistort.

        Parameters
        ----------
        niter : int, optional
            Number of basin-hopping iterations to perform (default is 500).
        stepsize : float, optional
            Maximum step size for random displacements in the rotation angles (default is 100 degrees).
        T : float, optional
            Temperature parameter for the Metropolis criterion (default is 10).
        disp : bool, optional
            If True, displays optimization progress in the console.

        Returns
        -------
        None

        Notes
        -----
        - Uses L-BFGS-B as the local minimizer with bounds on rotation angles [0, 360] degrees.
        - Updates the internal attributes:
            `optimal_xaxis_rotation`, `optimal_yaxis_rotation`, `optimal_zaxis_rotation`,
            and `rdistort_value`.
        """

        # Wrapper for the objective function
        def objective(x):
            theta, phi, psi = x
            return self.Rotate_and_Calculate_rdistort_value(theta, phi, psi)

        # Optional: bounds or constraints (basinhopping supports them via 'minimizer_kwargs')
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

    def kabsch_alignment(self, reference_vectors, test_vectors):
        # Written by ChatGPT
        """Align test_vectors to reference_vectors using the Kabsch algorithm."""
        C = np.dot(reference_vectors.T, test_vectors)
        V, S, W = np.linalg.svd(C)
        d = np.sign(np.linalg.det(np.dot(W.T, V.T)))
        rotation_matrix = np.dot(W.T, np.dot(np.diag([1, 1, d]), V.T))
        return rotation_matrix

    def Minimize_rdistort_KabschAlignment(self):
        # Generate all possible permutations of the test vector set
        permutations = list(itertools.permutations(self.test_vector_set))
        # Calculate the rdistort value for each permutation using Kabsch alignment
        smallest_rdistort_value = float("inf")
        count = 0
        for permutation_vectors in permutations:
            # Calculate the rotation matrix using Kabsch alignment
            rotation_matrix = self.kabsch_alignment(
                np.array(self.reference_vector_set), np.array(permutation_vectors)
            )
            rotated_vectors = [np.dot(rotation_matrix, v) for v in permutation_vectors]
            # Calculate the rdistort value for the rotated vectors
            rdistort_value = self.Calculate_rdistort_value_NewTest_vector_set(
                rotated_vectors
            )
            if rdistort_value < smallest_rdistort_value:
                smallest_rdistort_value = rdistort_value
                self.test_vector_set = rotated_vectors

    def Minimize_rdistort_BruteForce(
        self,
        grid_size=10,
        x_angles_range: list = [0, 360],
        y_angles_range: list = [0, 360],
        z_angles_range: list = [0, 360],
    ):
        """
        Brute-force search to minimize the rdistort value by testing all combinations of Euler angles.

        Parameters
        ----------
        grid_size : int, optional
            Number of discrete steps for each rotation angle (default is 10).

        Returns
        -------
        None

        Notes
        -----
        - This method exhaustively tests all combinations of Euler angles in the range [0, 360] degrees.
        - Updates the internal attributes:
            `optimal_xaxis_rotation`, `optimal_yaxis_rotation`, `optimal_zaxis_rotation`,
            and `rdistort_value`.
        """
        x_angles = np.arange(x_angles_range[0], x_angles_range[1], grid_size)
        y_angles = np.arange(y_angles_range[0], y_angles_range[1], grid_size)
        z_angles = np.arange(z_angles_range[0], z_angles_range[1], grid_size)
        best_rdistort_value = float("inf")
        best_angles = (0, 0, 0)

        for theta in x_angles:
            for phi in y_angles:
                for psi in z_angles:
                    rdistort_value = self.Rotate_and_Calculate_rdistort_value(
                        theta, phi, psi
                    )
                    if rdistort_value < best_rdistort_value:
                        best_rdistort_value = rdistort_value
                        best_angles = (theta, phi, psi)

        (
            self.optimal_xaxis_rotation,
            self.optimal_yaxis_rotation,
            self.optimal_zaxis_rotation,
        ) = best_angles
        self.rdistort_value = best_rdistort_value

    def Minimize_rdistort_EfficientBruteForce(
        self,
        grid_size_first_stage=10,
        grid_size_second_stage=1,
    ):
        self.Minimize_rdistort_BruteForce(
            grid_size=grid_size_first_stage,
        )
        self.Minimize_rdistort_BruteForce(
            grid_size=grid_size_second_stage,
            x_angles_range=[
                self.optimal_xaxis_rotation - grid_size_first_stage / 2,
                self.optimal_xaxis_rotation + grid_size_first_stage / 2,
            ],
            y_angles_range=[
                self.optimal_yaxis_rotation - grid_size_first_stage / 2,
                self.optimal_yaxis_rotation + grid_size_first_stage / 2,
            ],
            z_angles_range=[
                self.optimal_zaxis_rotation - grid_size_first_stage / 2,
                self.optimal_zaxis_rotation + grid_size_first_stage / 2,
            ],
        )
        self.Minimize_rdistort_BasisHopping(
            niter=100,
            stepsize=grid_size_second_stage,
            T=1,
            x0=[
                self.optimal_xaxis_rotation,
                self.optimal_yaxis_rotation,
                self.optimal_zaxis_rotation,
            ],
            bounds=[
                (
                    self.optimal_xaxis_rotation - grid_size_second_stage,
                    self.optimal_xaxis_rotation + grid_size_second_stage,
                ),
                (
                    self.optimal_yaxis_rotation - grid_size_second_stage,
                    self.optimal_yaxis_rotation + grid_size_second_stage,
                ),
                (
                    self.optimal_zaxis_rotation - grid_size_second_stage,
                    self.optimal_zaxis_rotation + grid_size_second_stage,
                ),
            ],
        )
