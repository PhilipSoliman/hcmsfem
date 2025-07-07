import os
import tempfile
import warnings
from typing import Optional

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import factorized, spsolve
from tqdm import tqdm

from hcmsfem.fespace import FESpace
from hcmsfem.logger import LOGGER, PROGRESS
from hcmsfem.meshes import TwoLevelMesh
from hcmsfem.problem_type import ProblemType
from hcmsfem.solvers.direct_sparse import DirectSparseSolver, MatrixType

warnings.simplefilter("ignore", SparseEfficiencyWarning)


class CoarseSpace(object):
    NAME = "Coarse space (base)"
    SHORT_NAME = "CS (base)"

    def __init__(
        self,
        A: sp.csr_matrix,
        fespace: FESpace,
        two_mesh: TwoLevelMesh,
        ptype: ProblemType = ProblemType.DIFFUSION,
        progress: Optional[PROGRESS] = None,  # just for type hinting
    ):
        self.fespace = fespace
        self.A = A
        self.two_mesh = two_mesh
        self.ptype = ptype
        LOGGER.debug(f"Initialized {self.NAME}")

    def assemble_coarse_operator(self, A: sp.csr_matrix) -> sp.csc_matrix:
        LOGGER.info("Assembling coarse operator")
        if not hasattr(self, "restriction_operator"):
            self.restriction_operator = self.assemble_restriction_operator()
        LOGGER.debug("Applying restriction operator to A")
        coarse_op = self.restriction_operator.transpose() @ (
            A @ self.restriction_operator
        )
        LOGGER.info("Obtained coarse operator")
        return coarse_op

    def assemble_restriction_operator(self) -> sp.csc_matrix:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_restriction_operator_bases(self) -> dict[str, ngs.GridFunction]:
        if (
            restriction_operator := getattr(self, "restriction_operator", None)
        ) is not None:
            try:
                num_bases = restriction_operator.shape[1]
                bases = {}
                for i in range(num_bases):
                    vals = np.zeros(self.fespace.total_dofs)
                    vals[self.fespace.free_dofs_mask] = (
                        restriction_operator[:, i].toarray().flatten()
                    )
                    gfunc = self.fespace.get_gridfunc(vals)
                    bases[f"coarse_basis_{i}"] = gfunc
                LOGGER.debug(
                    f"Restriction operator bases assembled: {len(bases)} bases"
                )
                return bases
            except MemoryError:
                msg = "MemoryError: The restriction operator bases are too large to fit in memory."
                LOGGER.warning(msg)
                return {}
        else:
            msg = "Restriction operator is not assembled yet."
            LOGGER.error(msg)
            raise ValueError(msg)

    def __str__(self):
        return self.NAME

    def _init_str(self):
        """
        Initialize string representation of the coarse space.
        """
        return (
            f"Coarse space info: \n\tfree dofs: {self.fespace.num_free_dofs}"
            f"{self._meta_info()}"
        )

    def _meta_info(self) -> str:
        """
        Return a string with meta information about the coarse space.
        This method can be overridden by subclasses to provide more specific information.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses to provide meta information."
        )


class Q1CoarseSpace(CoarseSpace):
    NAME = "Q1 coarse space"
    SHORT_NAME = "Q1"

    def __init__(
        self,
        A: sp.csr_matrix,
        fespace: FESpace,
        two_mesh: TwoLevelMesh,
        progress: Optional[PROGRESS] = None,
    ):
        super().__init__(A, fespace, two_mesh)
        self.restriction_operator = self.assemble_restriction_operator()
        self.coarse_dimension = self.restriction_operator.shape[1]  # type: ignore
        LOGGER.info(f"{self} initialized")
        LOGGER.debug(str(self._init_str()))

    def assemble_restriction_operator(self) -> sp.csc_matrix:
        prolongation_operator = self.fespace.get_prolongation_operator()
        restrict_operator = prolongation_operator[self.fespace.free_dofs_mask, :]
        LOGGER.debug("Restriction operator assembled")
        return restrict_operator

    def _meta_info(self) -> str:
        return f"\n\tcoarse dimension: {self.coarse_dimension}"


class GDSWCoarseSpace(CoarseSpace):
    NAME = "GDSW coarse space"
    SHORT_NAME = "GDSW"

    def __init__(
        self,
        A: sp.csr_matrix,
        fespace: FESpace,
        two_mesh: TwoLevelMesh,
        progress: Optional[PROGRESS] = None,
    ):
        self.progress = PROGRESS.get_active_progress_bar(progress)
        task = self.progress.add_task(f"Initializing {self.NAME}", total=3)
        LOGGER.info(f"Initializing {self.NAME}")

        super().__init__(A, fespace, two_mesh)
        self.progress.advance(task)

        # null space basis and dimension
        self.null_space_dim = self._get_null_space_basis()
        self.progress.advance(task)

        # all connected components
        self.num_coarse_node_components = len(fespace.free_coarse_node_dofs)
        self.num_edge_components = len(fespace.free_edge_component_dofs)
        self.num_face_components = len(fespace.free_face_component_dofs)
        self.interface_components = self._get_connected_components()
        self.progress.advance(task)

        # interface component dimension
        self.interface_dimension = len(self.interface_components) * self.null_space_dim

        LOGGER.info(f"{self} initialized")
        LOGGER.debug(str(self._init_str()))
        self.progress.soft_stop()

    def assemble_restriction_operator(self) -> sp.csc_matrix:
        LOGGER.debug(f"Assembling restriction operator for {self.NAME}")

        # get the interface operator
        interface_operator = self._assemble_interface_operator()

        # create the restriction operator
        restriction_operator = sp.csc_matrix(
            (self.fespace.num_free_dofs, self.interface_dimension), dtype=float
        )

        # interior <- interior matrix
        A_II = self.A[self.fespace.interior_dofs_mask, :][
            :, ~self.fespace.interface_dofs_mask
        ]
        LOGGER.debug("Assembled interior <- interior matrix A_II")

        # interior <- interface matrix
        A_IGamma = self.A[self.fespace.interior_dofs_mask, :][
            :, self.fespace.interface_dofs_mask
        ]
        LOGGER.debug("Assembled interior <- interface matrix A_IGamma")

        # discrete harmonic extension
        A_II_csc = A_II.tocsc()
        sparse_solver = DirectSparseSolver(
            A_II_csc,
            matrix_type=MatrixType.SPD,
            multithreaded=False,
            progress=self.progress,
        )
        rhs = (A_IGamma @ interface_operator).tocsc()
        interior_op = -sparse_solver(rhs)
        LOGGER.debug("Got interior operator via discrete harmonic extension")

        # Efficiently stack the operators
        blocks = [
            sp.csc_matrix(interior_op),  # shape: (num_interior, interface_dim)
            sp.csc_matrix(interface_operator),  # shape: (num_interface, interface_dim)
        ]
        stacked = sp.vstack(blocks, format="csc")
        LOGGER.debug("Stacked interior and interface operators")

        # Now, permute rows to match the original free DOF ordering
        # Create a permutation array: indices of [interior_dofs, interface_dofs] in the free_dofs_mask
        interior_idx = np.where(~self.fespace.interface_dofs_mask)[0]
        interface_idx = np.where(self.fespace.interface_dofs_mask)[0]
        perm = np.argsort(np.concatenate([interior_idx, interface_idx]))
        LOGGER.debug("Created permutation array for restriction operator")

        # Apply the permutation to the stacked operator
        restriction_operator = stacked[perm, :]  # type: ignore
        LOGGER.debug("Obtained restriction operator by applying permutation")

        return restriction_operator

    def _assemble_interface_operator(self):
        """
        Assemble the interface operator for the GDSW coarse space.
        Note: for problems with multiple dimensions, coordinate dofs corresponding to one node are spaced by
        ndofs (so if ndof = 8, then dofs for node 0 are 0, 8, 16, ...).
        """
        progress = PROGRESS.get_active_progress_bar(self.progress)
        task = progress.add_task(
            "Assembling interface operator",
            total=len(self.interface_components),
        )
        interface_index = 0
        entries = ([], ([], []))  # data, (rows, cols)
        for interface_component in self.interface_components:
            self._restrict_null_space_to_interface_component(
                entries, interface_component, interface_index
            )

            # update the interface index for the next interface component with the null space dimension
            interface_index += self.null_space_dim

            progress.advance(task)

        interface_operator = sp.coo_array(
            entries,
            shape=(self.fespace.total_dofs, self.interface_dimension),
            dtype=float,
        ).tocsc()

        # restrict the interface operator to the free and interface dofs
        interface_operator = interface_operator[self.fespace.free_dofs_mask, :][
            self.fespace.interface_dofs_mask, :
        ]

        LOGGER.debug("Assembled interface operator")
        progress.soft_stop()
        return interface_operator

    def _restrict_null_space_to_interface_component(
        self,
        entries: tuple[list[float], tuple[list[int], list[int]]],  # data, (rows, cols)
        interface_component: np.ndarray,
        interface_index: int,
        partition_of_unity: int | np.ndarray = 1,
    ):
        """
        Restrict the null space to the interface operator for the given interface component and for all problem coordinates.
        """
        # unpack the entries
        data, (rows, cols) = entries

        # indices of all dofs
        ndofs = self.fespace.total_dofs
        ndofs_prev = 0
        for coord, ndofs in enumerate(self.fespace.ndofs_per_unknown):
            # NOTE: NGSolve stores coordinate dofs corresponding to one node spaced by ndofs / dimension
            idxs = np.arange(ndofs_prev, ndofs_prev + ndofs)

            # get dofs for current coordinate coord
            coord_idxs_mask = np.logical_and(
                ndofs_prev < interface_component,
                interface_component < ndofs_prev + ndofs,
            )

            # get indices of dofs for the current coordinate
            component_coord_idxs = interface_component[coord_idxs_mask].tolist()
            rows_coord = component_coord_idxs * self.null_space_dim
            cols_coord = []
            data_coord = []
            for i in range(self.null_space_dim):
                cols_coord.extend([interface_index + i for _ in component_coord_idxs])
                data_coord.extend(
                    [
                        self.null_space_basis[coord, i] * partition_of_unity
                        for _ in component_coord_idxs
                    ]
                )
            rows.extend(rows_coord)
            cols.extend(cols_coord)
            data.extend(data_coord)

            # increment the previous dofs index
            ndofs_prev += ndofs

    def _get_null_space_basis(self):
        self.null_space_basis = np.array([])
        null_space_dim = 0
        if self.ptype == ProblemType.DIFFUSION:
            self.null_space_basis = np.ones((1, 1))
            null_space_dim = 1
        elif self.ptype == ProblemType.NAVIER_STOKES:
            raise NotImplementedError(
                "N.S. problem type not implemented for coarse space."
            )
        else:
            msg = f"Unsupported problem type: {self.ptype}"
            LOGGER.error(msg)
            raise ValueError(msg)
        return null_space_dim

    def _get_connected_components(self):
        interface_components = []
        # coarse node components
        if len(self.fespace.free_coarse_node_dofs) > 0:
            interface_components.extend(self._get_coarse_components())

        # edge components
        if len(self.fespace.free_edge_component_dofs) > 0:
            interface_components.extend(self._get_edge_components())

        # face components
        if len(self.fespace.free_face_component_dofs) > 0:
            interface_components.extend(self._get_face_components())

        LOGGER.debug(
            f"Obtained {len(interface_components)} interface components from coarse space"
        )
        return interface_components

    def _get_coarse_components(self) -> list[np.ndarray]:
        components = []
        for coarse_component_dofs in self.fespace.free_coarse_node_dofs:
            components.append(np.array(coarse_component_dofs))
        return components

    def _get_edge_components(self) -> list[np.ndarray]:
        components = []
        for edge_component_dofs in self.fespace.free_edge_component_dofs:
            components.append(np.array(edge_component_dofs))
        return components

    def _get_face_components(self) -> list[np.ndarray]:
        components = []
        for face_component_dofs in self.fespace.free_face_component_dofs:
            components.append(np.array(face_component_dofs))
        return components

    def _meta_info(self) -> str:
        return (
            f"\n\tnull space dim: {self.null_space_dim}"
            f"\n\tinterface dim: {self.interface_dimension}"
            f"\n\t\tcoarse node components: {self.num_coarse_node_components}"
            f"\n\t\tedge components: {self.num_edge_components}"
            f"\n\t\tface components: {self.num_face_components}"
        )


class RGDSWCoarseSpace(GDSWCoarseSpace):
    NAME = "RGDSW coarse space"
    SHORT_NAME = "RGDSW"

    def __init__(
        self,
        A: sp.csr_matrix,
        fespace: FESpace,
        two_mesh: TwoLevelMesh,
        progress: Optional[PROGRESS] = None,
    ):
        self.progress = PROGRESS.get_active_progress_bar(progress)
        task = self.progress.add_task(f"Initializing {self.NAME}", total=3)
        LOGGER.info(f"Initializing {self.NAME}")

        CoarseSpace.__init__(self, A, fespace, two_mesh)
        self.progress.advance(task)

        # null space basis and dimension
        self.null_space_dim = self._get_null_space_basis()
        self.progress.advance(task)

        # interface components
        self.component_tree_dofs = self.fespace.free_component_tree_dofs
        self.edge_component_multiplicities = self.fespace.edge_component_multiplicities
        self.progress.advance(task)

        # interface dimension
        self.interface_dimension = len(self.component_tree_dofs) * self.null_space_dim
        if self.interface_dimension == 0:
            raise ValueError("No RGDSW components with coarse node ancestors found")

        LOGGER.info(f"{self} initialized")
        LOGGER.debug(str(self._init_str()))

        self.progress.soft_stop()

    def _assemble_interface_operator(self):
        progress = PROGRESS.get_active_progress_bar(self.progress)
        task = progress.add_task(
            "Assembling interface operator",
            total=len(self.component_tree_dofs),
        )

        entries = ([], ([], []))  # data, (rows, cols)
        interface_index = 0
        for component_dofs in self.component_tree_dofs.values():
            node_dofs = component_dofs["node"]
            edge_components = component_dofs["edges"]
            self._restrict_null_space_to_interface_component(
                entries, np.array(node_dofs), interface_index
            )

            for edge, edge_dofs in edge_components.items():
                multiplicity = self.edge_component_multiplicities[edge]
                self._restrict_null_space_to_interface_component(
                    entries, np.array(edge_dofs), interface_index, 1 / multiplicity
                )

            # update the interface index for the next interface component with the null space dimension
            interface_index += self.null_space_dim

            progress.advance(task)

        interface_operator = sp.coo_array(
            entries,
            shape=(self.fespace.total_dofs, self.interface_dimension),
            dtype=float,
        ).tocsc()

        # restric the interface operator to the free and interface dofs
        interface_operator = interface_operator[self.fespace.free_dofs_mask, :][
            self.fespace.interface_dofs_mask, :
        ]

        LOGGER.debug("Assembled interface operator")
        progress.soft_stop()
        return interface_operator

    def _meta_info(self) -> str:
        return (
            f"\n\tnull space dim: {self.null_space_dim}"
            f"\n\tinterface dim: {self.interface_dimension}"
        )


class AMSCoarseSpace(GDSWCoarseSpace):
    NAME = "AMS coarse space"
    SHORT_NAME = "AMS"

    def __init__(
        self,
        A: sp.csr_matrix,
        fespace: FESpace,
        two_mesh: TwoLevelMesh,
        progress: Optional[PROGRESS] = None,
        mesh_size_threshold: float = 0,  # threshold for large meshes
    ):
        self.progress = PROGRESS.get_active_progress_bar(progress)
        task = self.progress.add_task(f"Initializing {self.NAME}", total=3)
        LOGGER.info(f"Initializing {self.NAME}")

        # base initialization
        CoarseSpace.__init__(self, A, fespace, two_mesh)

        # get dofs
        self.progress.advance(task)

        self.coarse_dofs, self.edge_dofs, self.face_dofs = (
            self._get_interface_component_masks()
        )
        self.progress.advance(task)
        self.interface_dimension = len(self.coarse_dofs)
        self.num_coarse_dofs = self.interface_dimension
        self.num_edge_dofs = len(self.edge_dofs)
        self.num_face_dofs = len(self.face_dofs)
        self.interior_dofs_mask = ~self.fespace.interface_dofs_mask
        self.num_interior_dofs = np.sum(self.interior_dofs_mask)
        self.progress.advance(task)

        # get sorted dofs for clipping edge and face restriction operators
        self.coarse_dofs_argsort = np.argsort(self.coarse_dofs)
        self.coarse_dofs_sorted = self.coarse_dofs[self.coarse_dofs_argsort]
        self.edge_dofs_argsort = np.argsort(self.edge_dofs)
        self.edge_dofs_sorted = self.edge_dofs[self.edge_dofs_argsort]

        # handle large meshes
        self.mesh_size_threshold = mesh_size_threshold
        if self.two_mesh.coarse_mesh_size <= self.mesh_size_threshold:
            self.batch_size = 16
            self.num_batches = (
                self.interface_dimension + self.batch_size - 1
            ) // self.batch_size

        LOGGER.info(f"{self} initialized")
        LOGGER.debug(str(self._init_str()))
        self.progress.soft_stop()

    def assemble_restriction_operator(self) -> sp.csc_matrix:
        LOGGER.debug(f"Assembling restriction operator for {self}")
        restriction_operator = sp.csc_matrix(
            (self.fespace.num_free_dofs, self.interface_dimension), dtype=float
        )

        # Phi_V
        vertex_restriction = sp.eye(self.interface_dimension, dtype=float).tocsc()
        LOGGER.debug("Assembled vertex restriction operator")

        # A_EE
        A_EE = self.A[self.edge_dofs, :][:, self.edge_dofs]
        A_EI = self.A[self.edge_dofs, :][:, self.fespace.interior_dofs_mask]
        A_EE += sp.diags(
            A_EI.sum(axis=1).A1, offsets=0, format="csc"
        )  # NOTE: SPD-ness is (possibly) lost here

        # free up memory
        A_EI = sp.csc_matrix(A_EI.shape, dtype=float)

        if np.any(self.face_dofs):
            A_EF = self.A[self.edge_dofs, :][:, self.face_dofs]
            A_EE += sp.diags(A_EF.sum(axis=1).A1, offsets=0, format="csc")

            # free up memory
            A_EF = sp.csc_matrix(A_EF.shape, dtype=float)
        LOGGER.debug("Assembled edge <- edge matrix A_EE")

        # get solver for edge restriction operator and free memory
        sparse_solver = DirectSparseSolver(
            A_EE.tocsc(),
            matrix_type=MatrixType.Symmetric,
            progress=self.progress,
            delete_rhs=True,
        )
        A_EE = sp.csc_matrix(A_EE.shape, dtype=float)

        # clip edge restriction operator to be non-zero only on their respective connected components
        task = self.progress.add_task(
            "Clipping edge restriction operator to coarse node support",
            total=len(self.fespace.free_component_tree_dofs) + 1,
        )

        # get all edge dofs that are in the support of coarse nodes
        edge_restriction_nzindices = [None] * self.num_coarse_dofs
        for dofs in self.fespace.free_component_tree_dofs.values():
            coarse_dofs = dofs["node"]

            # restrict coarse dofs to the free coarse dofs
            coarse_dofs = self.fespace.map_global_to_restricted_dofs(
                np.array(coarse_dofs)
            )

            # get edge dofs in the support of the coarse node
            edge_dofs = []
            for edge in dofs["edges"].values():
                edge_dofs.extend(edge)

            # restrict edge dofs to the free edge dofs
            edge_dofs = self.fespace.map_global_to_restricted_dofs(np.array(edge_dofs))

            # fill the matrix by looping of coarse dofs
            for coarse_dof in coarse_dofs:
                # get coarse dof index in list of coarse dofs
                coarse_idx = np.searchsorted(self.coarse_dofs_sorted, coarse_dof)
                coarse_idx = self.coarse_dofs_argsort[coarse_idx]

                # get edge indices in the list of edge dofs
                edge_idxs = np.searchsorted(self.edge_dofs_sorted, edge_dofs)
                edge_idxs = self.edge_dofs_argsort[edge_idxs]

                # store non-zero indices for the edge restriction operator
                edge_restriction_nzindices[coarse_idx] = edge_idxs
            self.progress.advance(task)

        self.progress.remove_task(task)
        LOGGER.debug("Filtered edge restriction operator to coarse node support")

        edge_restriction = -sparse_solver(
            self.A[self.edge_dofs, :][:, self.coarse_dofs],
            nzindices=edge_restriction_nzindices,
        )
        sparse_solver = None  # free up memory
        LOGGER.debug("Assembled edge restriction operator")

        # Phi_F #TODO: probably need to do similar filtering to A_EV for face dofs
        face_restriction = sp.csc_matrix(
            (self.num_face_dofs, self.interface_dimension), dtype=float
        )
        if np.any(self.face_dofs):
            A_FF = self.A[self.face_dofs, :][:, self.face_dofs]
            A_FI = self.A[self.face_dofs, :][
                :, self.fespace.interior_dofs_mask
            ]  # NOTE: SPD-ness is lost here
            A_FF += sp.diags(A_FI.sum(axis=1).A1, offsets=0, format="csc")
            A_FV = self.A[self.face_dofs, :][:, self.coarse_dofs]
            A_FE = self.A[self.face_dofs, :][:, self.edge_dofs]
            sparse_solver = DirectSparseSolver(
                A_FF.tocsc(), matrix_type=MatrixType.Symmetric, progress=self.progress
            )
            face_restriction = -sparse_solver(
                (A_FV @ vertex_restriction + A_FE @ edge_restriction).tocsc()
            )
            LOGGER.debug("Assembled face restriction operator")

        # Phi_I
        LOGGER.debug("Assembling interior restriction operator")
        A_II = self.A[self.interior_dofs_mask, :][:, self.interior_dofs_mask]
        A_IV = self.A[self.interior_dofs_mask, :][:, self.coarse_dofs]
        A_IE = self.A[self.interior_dofs_mask, :][:, self.edge_dofs]
        A_IF = self.A[self.interior_dofs_mask, :][:, self.face_dofs]
        if self.two_mesh.coarse_mesh_size > self.mesh_size_threshold:
            # Phi_Gamma
            interface_restriction = (
                A_IV @ vertex_restriction
                + A_IE @ edge_restriction
                + A_IF @ face_restriction
            ).tocsc()

            # instantiate the sparse solver
            sparse_solver = DirectSparseSolver(
                A_II.tocsc(),
                matrix_type=MatrixType.SPD,
                progress=self.progress,
                delete_rhs=True,
            )

            # free up memory
            A_II = sp.csc_matrix(A_II.shape, dtype=float)
            A_IV = sp.csc_matrix(A_IV.shape, dtype=float)
            A_IE = sp.csc_matrix(A_IE.shape, dtype=float)
            A_IF = sp.csc_matrix(A_IF.shape, dtype=float)

            LOGGER.debug("Solving full interface restriction")
            interior_restriction = -sparse_solver(interface_restriction)

            sparse_solver = None  # free up memory
        else:
            # get solver
            sparse_solver = DirectSparseSolver(
                A_II.tocsc(),
                matrix_type=MatrixType.Symmetric,
                progress=self.progress,
                batch_size=self.batch_size,
                delete_rhs=True,
            )

            # free up memory
            A_II = sp.csc_matrix(A_II.shape, dtype=float)

            # set log level to WARNING to avoid too many debug messages
            progress = PROGRESS.get_active_progress_bar(self.progress)
            loglevel = LOGGER.level

            LOGGER.debug("Iteratively solving interface restriction columns")
            task = progress.add_task(
                "Solving interface restriction columns",
                total=self.num_batches,
            )
            LOGGER.setLevel(LOGGER.WARNING)
            cols = []
            with tempfile.TemporaryDirectory() as tempdir:
                for b in range(self.num_batches):
                    # get start and end indices for the current batch
                    start_idx = b * self.batch_size
                    end_idx = min((b + 1) * self.batch_size, self.interface_dimension)

                    # get the current batch of interface restriction columns
                    interface_restriction_col = (
                        A_IV @ vertex_restriction[:, start_idx:end_idx]
                        + A_IE @ edge_restriction[:, start_idx:end_idx]
                        + A_IF @ face_restriction[:, start_idx:end_idx]
                    ).tocsc()

                    fp = os.path.join(tempdir, f"interface_restriction_col_{b}.npz")
                    with open(fp, "wb") as temp_file:
                        sp.save_npz(
                            temp_file,
                            -sparse_solver(interface_restriction_col),
                            compressed=True,
                        )

                    progress.advance(task)
                progress.remove_task(task)

                # restore log level
                LOGGER.setLevel(loglevel)

                # free up memory
                A_IV = sp.csc_matrix(A_IV.shape, dtype=float)
                A_IE = sp.csc_matrix(A_IE.shape, dtype=float)
                A_IF = sp.csc_matrix(A_IF.shape, dtype=float)
                sparse_solver = None

                # load data from the temporary file
                task = progress.add_task(
                    "Loading interface restriction columns from temporary file",
                    total=self.num_batches,
                )
                LOGGER.debug(
                    "Loading interior restriction columns from temporary files"
                )
                cols = []
                for b in range(self.num_batches):
                    fp = os.path.join(tempdir, f"interface_restriction_col_{b}.npz")
                    with open(fp, "rb") as temp_file:
                        cols.append(sp.load_npz(temp_file).tocsc())
                    progress.advance(task)

                LOGGER.debug("Stacking interior restriction columns")
                interior_restriction = sp.hstack(cols, format="csc")

                # free up memory
                del cols

            # stop progress bar
            progress.soft_stop()

        # set interface restriction to empty matrix to save memory
        interface_restriction = sp.csc_matrix(
            (self.num_interior_dofs, self.interface_dimension), dtype=float
        )
        LOGGER.debug("Assembled interior restriction operator")

        # Efficiently stack the operators in the order: coarse, edge, face, interior
        blocks = [
            vertex_restriction,  # shape: (num_coarse_dofs, interface_dim)
            edge_restriction,  # shape: (num_edge_dofs, interface_dim)
            face_restriction,  # shape: (num_face_dofs, interface_dim)
            interior_restriction,  # shape: (num_interior_dofs, interface_dim)
        ]
        stacked = sp.vstack(blocks, format="csc")

        # free up memory
        del blocks
        LOGGER.debug(
            f"Stacked restriction operators for coarse, edge, {'face,' if np.any(self.face_dofs) else ''} and interior"
        )

        # free up memory
        vertex_restriction = sp.csc_matrix(
            (self.num_coarse_dofs, self.interface_dimension), dtype=float
        )
        edge_restriction = sp.csc_matrix(
            (self.num_edge_dofs, self.interface_dimension), dtype=float
        )
        face_restriction = sp.csc_matrix(
            (self.num_face_dofs, self.interface_dimension), dtype=float
        )
        interior_restriction = sp.csc_matrix(
            (self.num_interior_dofs, self.interface_dimension), dtype=float
        )

        # Build the permutation array to match the original free DOF ordering
        coarse_idx = np.array(self.coarse_dofs)
        edge_idx = np.array(self.edge_dofs)
        face_idx = np.array(self.face_dofs)
        interior_idx = np.where(self.interior_dofs_mask)[0]
        perm = np.argsort(
            np.concatenate([coarse_idx, edge_idx, face_idx, interior_idx])
        )
        LOGGER.debug("Created permutation array for restriction operator")

        # Apply the permutation to the stacked operator
        restriction_operator = stacked[perm, :]  # type: ignore
        LOGGER.debug("Obtained restriction operator by applying permutation")

        return restriction_operator

    def _get_interface_component_masks(self):
        # get all groups of interface components
        coarse_components = []
        for coarse_component in super()._get_coarse_components():
            free_dofs = self.fespace.map_global_to_restricted_dofs(coarse_component)
            coarse_components.extend(free_dofs.tolist())

        edge_components = []
        for edge_component in super()._get_edge_components():
            free_dofs = self.fespace.map_global_to_restricted_dofs(edge_component)
            edge_components.extend(free_dofs.tolist())

        face_components = []
        for face_component in super()._get_face_components():
            free_dofs = self.fespace.map_global_to_restricted_dofs(face_component)
            face_components.extend(free_dofs.tolist())

        LOGGER.debug(
            f"Obtained {len(coarse_components)} coarse, {len(edge_components)} edge, and {len(face_components)} face components"
        )
        return (
            np.array(coarse_components),
            np.array(edge_components),
            np.array(face_components),
        )

    def _meta_info(self) -> str:
        return (
            f"\n\tinterface dimension: {self.interface_dimension}"
            f"\n\tcoarse dofs: {self.num_coarse_dofs}"
            f"\n\tedge dofs: {self.num_edge_dofs}"
            f"\n\tface dofs: {self.num_face_dofs}"
        )
