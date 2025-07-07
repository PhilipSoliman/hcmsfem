from types import NoneType
from typing import Callable, Optional, Type

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import LinearOperator, factorized, spsolve

from hcmsfem import gpu_interface as gpu
from hcmsfem.fespace import FESpace
from hcmsfem.logger import LOGGER, PROGRESS
from hcmsfem.meshes import TwoLevelMesh
from hcmsfem.operators import Operator
from hcmsfem.preconditioners import CoarseSpace
from hcmsfem.solvers import DirectSparseSolver, MatrixType


class OneLevelSchwarzPreconditioner(Operator):
    NAME = "1-level Schwarz preconditioner"
    SHORT_NAME = "1-OAS"

    def __init__(
        self,
        A: sp.csr_matrix,
        fespace: FESpace,
        use_gpu: bool = False,
        progress: Optional[PROGRESS] = None,
    ):
        self.progress = PROGRESS.get_active_progress_bar(progress)
        task = self.progress.add_task(
            f"Initializing 1-level Schwarz preconditioner", total=3
        )
        LOGGER.info("Initializing 1-level Schwarz preconditioner")

        self.shape = A.shape
        self.fespace = fespace
        self.use_gpu = use_gpu
        self.local_free_dofs = self._get_local_free_dofs()
        self.progress.advance(task)
        self.local_operators = self._get_local_operators(A)
        self.progress.advance(task)
        self.local_solvers = self._get_local_solvers()
        self.progress.advance(task)

        LOGGER.info("1-level Schwarz preconditioner initialized")
        self.progress.soft_stop()

    def apply(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        for dofs, solver in zip(self.local_free_dofs, self.local_solvers):
            out[dofs] += solver(x[dofs])  # type: ignore
        return out

    def apply_gpu(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x, dtype=torch.float64)
        for dofs, solver in zip(self.local_free_dofs, self.local_solvers):
            tmp = torch.zeros_like(x[dofs], dtype=torch.float64)
            solver(x[dofs], tmp)  # type: ignore
            out[dofs] += tmp
        return out

    def as_linear_operator(self) -> LinearOperator:
        """Return the preconditioner as a linear operator."""
        return LinearOperator(self.shape, lambda x: self.apply(x))

    def as_full_system(self, solve_gpu: bool = True) -> sp.csc_matrix:
        Minv = sp.lil_matrix(self.shape, dtype=float)
        if solve_gpu:
            LOGGER.debug("Returning preconditioner as full system (GPU)")
            for dofs, operator in zip(self.local_free_dofs, self.local_operators):
                solver = DirectSparseSolver(
                    operator, matrix_type=MatrixType.SPD, progress=self.progress
                )
                eye = sp.eye(operator.shape[0], dtype=float).tocsc()
                Minv[np.ix_(dofs, dofs)] += solver(eye)
        else:
            LOGGER.debug("Returning preconditioner as full system (CPU)")
            for dofs, operator in zip(self.local_free_dofs, self.local_operators):
                Minv[np.ix_(dofs, dofs)] += spsolve(operator, sp.eye(operator.shape[0]))  # type: ignore
        Minv = Minv.reshape(self.shape).tocsc()
        return Minv

    def _get_local_free_dofs(self) -> list[np.ndarray]:
        """Get local free dofs for each subdomain."""
        local_free_dofs = []
        for subdomain_dofs in self.fespace.domain_dofs.values():
            local_free_dofs.append(
                self.fespace.map_global_to_restricted_dofs(np.array(subdomain_dofs))
            )
        LOGGER.debug(f"Obtained local free dofs for {len(local_free_dofs)} subdomains")
        return local_free_dofs

    def _get_local_operators(self, A: sp.csr_matrix) -> list[sp.csc_matrix]:
        """Get local solvers for each subdomain."""
        local_operators = []
        task = self.progress.add_task(
            "Obtaining local operators", total=len(self.local_free_dofs)
        )
        for dofs in self.local_free_dofs:
            operator = A[dofs, :][:, dofs].tocsc()
            local_operators.append(operator)
            self.progress.advance(task)
        LOGGER.debug(f"Obtained local operators for {len(local_operators)} subdomains")
        self.progress.remove_task(task)
        return local_operators

    def _get_local_solvers(
        self,
    ) -> list[
        Callable[[np.ndarray], np.ndarray]
        | Callable[[torch.Tensor, torch.Tensor], NoneType]
    ]:
        """Get local solvers for each subdomain."""
        local_solvers = []
        task = self.progress.add_task(
            "Obtaining local solvers", total=len(self.local_operators)
        )
        for operator in self.local_operators:
            if not self.use_gpu:
                solver_f = factorized(operator)
            else:
                solver = DirectSparseSolver(operator, matrix_type=MatrixType.SPD).solver
                solver_f = lambda rhs, out: solver.solve(rhs, out)  # type: ignore
            local_solvers.append(solver_f)
            self.progress.advance(task)
        LOGGER.debug(
            f"Obtained local solvers for {len(local_solvers)} subdomains on {'GPU' if self.use_gpu else 'CPU'}"
        )
        self.progress.remove_task(task)
        return local_solvers

    def __str__(self):
        return self.NAME


class TwoLevelSchwarzPreconditioner(OneLevelSchwarzPreconditioner):
    NAME = "2-level Schwarz preconditioner"
    SHORT_NAME = "2-OAS"

    def __init__(
        self,
        A: sp.csr_matrix,
        fespace: FESpace,
        two_mesh: TwoLevelMesh,
        coarse_space: Type[CoarseSpace],
        coarse_only: bool = False,  # only use coarse space, skip 1-level preconditioner
        use_gpu: bool = False,  # construct preconditioner on GPU
        progress: Optional[PROGRESS] = None,
    ):
        self.progress = PROGRESS.get_active_progress_bar(progress)
        task = self.progress.add_task(
            "Initializing 2-level Schwarz preconditioner",
            total=3 - coarse_only + use_gpu,
        )

        self.coarse_only = coarse_only
        if not coarse_only:
            super().__init__(A, fespace, use_gpu, progress)
            self.progress.advance(task)
        else:
            self.shape = A.shape
            self.fespace = fespace
            self.use_gpu = use_gpu

        # get coarse space
        self.coarse_space = coarse_space(A, fespace, two_mesh, progress=self.progress)
        self.progress.advance(task)

        # get coarse operator
        self.coarse_op = self.coarse_space.assemble_coarse_operator(A)
        self.progress.advance(task)
        if use_gpu:
            self.restriction_operator = gpu.send_matrix(self.coarse_space.restriction_operator)  # type: ignore
            self.restriction_operator_T = gpu.send_matrix(self.coarse_space.restriction_operator.transpose())  # type: ignore
            LOGGER.info("Restriction operators sent to GPU")
            self.progress.advance(task)

        # get coarse solver
        self.coarse_solver = self._get_coarse_solver()
        self.progress.advance(task)

        # set preconditioner name
        self.NAME += f" with {self.coarse_space}"
        self.SHORT_NAME += f"-{self.coarse_space.SHORT_NAME}"

        LOGGER.info(f"{self} initialized")
        self.progress.soft_stop()

    def apply(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        if not self.coarse_only:
            out += super().apply(x)
        x_coarse = self.coarse_space.restriction_operator.transpose() @ x
        x_coarse = self.coarse_solver(x_coarse)  # type: ignore
        out += self.coarse_space.restriction_operator @ x_coarse

        return out

    def apply_gpu(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x, dtype=torch.float64)
        if not self.coarse_only:
            out += super().apply_gpu(x)
        x_coarse = torch.mv(self.restriction_operator_T, x)
        tmp = torch.zeros_like(x_coarse, dtype=torch.float64)
        self.coarse_solver(x_coarse, tmp)  # type: ignore
        out += torch.mv(self.restriction_operator, tmp)  # type: ignore
        return out

    def as_full_system(self) -> sp.csc_matrix:
        MinvA = sp.csc_matrix(self.shape, dtype=float)
        if not self.coarse_only:
            MinvA = super().as_full_system()
        solver = DirectSparseSolver(
            self.coarse_op, matrix_type=MatrixType.SPD, progress=self.progress
        )
        eye = sp.eye(
            self.coarse_space.restriction_operator.shape[0], dtype=float
        ).tocsc()
        MinvA = MinvA + self.coarse_space.restriction_operator @ solver(
            self.coarse_space.restriction_operator.transpose() @ eye
        )
        return MinvA

    def _get_coarse_solver(
        self,
    ) -> (
        Callable[[np.ndarray], np.ndarray]
        | Callable[[torch.Tensor, torch.Tensor], NoneType]
    ):
        if not self.use_gpu:
            LOGGER.debug("Obtained coarse solver (CPU)")
            return factorized(self.coarse_op.tocsc())
        else:
            with LOGGER.setLevelContext(LOGGER.WARNING):
                solver = DirectSparseSolver(
                    self.coarse_op, matrix_type=MatrixType.SPD
                ).solver
            solver_f = lambda rhs, out: solver.solve(rhs, out)  # type: ignore
            LOGGER.debug("Obtained coarse solver (GPU)")
            return solver_f

    def get_restriction_operator_bases(self) -> dict[str, ngs.GridFunction]:
        return self.coarse_space.get_restriction_operator_bases()
