import concurrent.futures
import multiprocessing
import threading
from enum import Enum
from typing import Callable, Optional

import cholespy as chol
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import eigsh, factorized, splu, spsolve
from tqdm import tqdm

from hcmsfem.gpu_interface import GPUInterface
from hcmsfem.logger import LOGGER, PROGRESS

# initialize GPU interface
gpu = GPUInterface()


class MatrixType(Enum):
    SPD = "SPD"
    Symmetric = "Symmetric"
    General = "General"


class DirectSparseSolver:
    NUM_CPU_THREADS = multiprocessing.cpu_count()
    GPU_BATCH_SIZE = 128
    CPU_BATCH_SIZE = 64
    DEVICE = gpu.DEVICE

    def __init__(
        self,
        A: sp.csc_matrix,
        matrix_type: Optional[MatrixType] = None,
        multithreaded: bool = False,
        progress: Optional[PROGRESS] = None,
        batch_size: Optional[int] = None,
        delete_rhs: bool = False,
    ):
        """
        Initialize a sparse solver. For SPD matrices a sparse Cholesky decomposition is made and efficiently
        reused for solving thereafter. If GPU is available, the sparse solve is performed on the GPU, otherwise on the CPU.
        For symmetric and general matrices, a scipy factorized solver is used.

        Multithreading, if enabled, is only used for symmetric and general matrices.

        If the matrix type is not provided, it will be inferred from the matrix properties.

        This solver is most efficient for medium-sized (O(n^3)-O(n^4)), sparse SPD matrices.

        Args:
            A (sp.csc_matrix): A symmetric matrix.
            matrix_type (Optional[MatrixType]): The type of the matrix (SPD, Symmetric, General).
            multithreaded (bool): Whether to use multithreaded (only relevant for Symmetric and General matrices).
        """
        LOGGER.debug("Initializing direct sparse solver")
        self.A = A
        self.matrix_type = matrix_type
        self.multithreaded = multithreaded
        self.progress = progress
        if batch_size is not None:
            self.batch_size = batch_size
        self.delete_rhs = delete_rhs
        self._solver = self.get_solver()
        LOGGER.debug("Initialized direct sparse solver")

    def __call__(
        self,
        rhs: sp.csc_matrix,
        nzindices: Optional[list] = None,
    ) -> sp.csc_matrix:
        """
        Multithreaded sparse solver for the interior operators.

        Args:
            A (sp.csc_matrix): A symmetric matrix.
            rhs (sp.csc_matrix): The right-hand side matrix.
            nzindices (Optional[list]): List of non-zero indices for each column of the right-hand side matrix.

        Returns:
            sp.csc_matrix: The solved interior operator as a sparse matrix.
        """
        if self.matrix_type == MatrixType.SPD:
            return self.cholesky(rhs, nzindices=nzindices)

        elif (
            self.matrix_type == MatrixType.Symmetric
            or self.matrix_type == MatrixType.General
        ):
            if self.multithreaded:
                return self.lu_threaded(rhs)
            else:
                return self.lu(rhs, nzindices=nzindices)
        else:
            raise ValueError(
                f"No direct solver available for matrix type {self.matrix_type}."
            )

    def get_solver(self) -> chol.CholeskySolverD | Callable[[np.ndarray], np.ndarray]:  # type: ignore
        LOGGER.debug(f"getting direct solver for {self.matrix_type} matrix")

        # select solver based on matrix type
        if self.matrix_type == MatrixType.SPD:
            LOGGER.debug(f"using cholesky solver")

            if gpu.AVAILABLE and not hasattr(self, "_batch_size"):
                self.batch_size = self.GPU_BATCH_SIZE
            elif not gpu.AVAILABLE and not hasattr(self, "_batch_size"):
                self.batch_size = self.CPU_BATCH_SIZE
            n = self.A.shape[0]  # type: ignore
            A_coo = self.A.tocoo()
            rows = torch.tensor(A_coo.row, dtype=torch.float64, device=self.DEVICE)
            cols = torch.tensor(A_coo.col, dtype=torch.float64, device=self.DEVICE)
            data = torch.tensor(A_coo.data, dtype=torch.float64, device=self.DEVICE)
            return chol.CholeskySolverD(n, rows, cols, data, chol.MatrixType.COO)  # type: ignore
        elif (
            self.matrix_type == MatrixType.Symmetric
            or self.matrix_type == MatrixType.General
        ):
            if not hasattr(self, "batch_size"):
                self.batch_size = self.CPU_BATCH_SIZE
            LOGGER.debug(f"using scipy factorized solver")
            return factorized(self.A)
        else:
            msg = f"Unsupported matrix type: {self.matrix_type}"
            LOGGER.error(msg)
            raise ValueError(msg)

    @property
    def solver(self) -> Callable[[np.ndarray], np.ndarray] | chol.CholeskySolverD:  # type: ignore
        return self._solver

    @property
    def matrix_type(self) -> MatrixType:
        return self._matrix_type

    @matrix_type.setter
    def matrix_type(self, matrix_type: Optional[MatrixType]) -> None:
        if matrix_type is None:
            if self.is_symmetric():
                if self.is_spd():
                    self._matrix_type = MatrixType.SPD
                else:
                    self._matrix_type = MatrixType.Symmetric
            else:
                self._matrix_type = MatrixType.General
        elif isinstance(matrix_type, MatrixType):
            self._matrix_type = matrix_type
        else:
            msg = f"Invalid matrix type: {matrix_type}. Must be None or an instance of MatrixType."
            LOGGER.error(msg)
            raise ValueError(msg)

    def is_symmetric(self, tol=1e-10) -> bool:
        # For sparse matrices, compare nonzero structure and values
        return (self.A - self.A.T).nnz == 0 and np.allclose(
            self.A.data, self.A.T.data, atol=tol
        )

    def is_spd(self, tol=1e-10) -> bool:
        try:
            # Compute the smallest algebraic eigenvalue
            vals = eigsh(self.A, k=1, which="SA", return_eigenvectors=False)
            return vals[0] > tol  # type: ignore
        except Exception:  # eigsh requires symmetric input
            return False

    def cholesky(
        self, rhs: sp.csc_matrix, nzindices: Optional[list] = None
    ) -> sp.csc_matrix:
        """
        Solve the system using Cholesky decomposition on GPU/CPU.

        Args:
            rhs (sp.csc_matrix): The right-hand side matrix.

        Returns:
            sp.csc_matrix: The solved interior operator as a sparse matrix.
        """
        LOGGER.debug("Solving using Cholesky decomposition")
        progress = PROGRESS.get_active_progress_bar(self.progress)
        n_rows, n_rhs = rhs.shape  # type: ignore
        num_batches = (n_rhs + self.batch_size - 1) // self.batch_size
        task = progress.add_task("Cholesky solving batches", total=num_batches)
        out_cols = []
        try:
            for i in range(num_batches):
                # get current batch
                if self.delete_rhs:
                    n_rhs = rhs.shape[1]  # type: ignore
                    start = 0
                    end = min(self.batch_size, n_rhs)
                    rhs_batch = rhs[:, start:end].tocoo()
                else:
                    start = i * self.batch_size
                    end = min((i + 1) * self.batch_size, n_rhs)
                    rhs_batch = rhs[:, start:end].tocoo()

                # rhs
                shape = (n_rows, end - start)
                rhs_batch_array = np.zeros(shape, dtype=np.float64)
                rhs_batch_array[rhs_batch.row, rhs_batch.col] = rhs_batch.data

                # send rhs to GPU
                rhs_device = gpu.send_array(rhs_batch_array, dtype=torch.float64)

                # output
                x = np.zeros_like(rhs_batch_array)
                x_device = gpu.send_array(x, dtype=torch.float64)

                # solve on GPU (if available) or CPU
                self.solver.solve(rhs_device, x_device)  # type: ignore

                # retrieve the result
                if nzindices is None:
                    x = gpu.retrieve_array(x_device).reshape(shape)
                elif isinstance(nzindices, list):
                    x = np.zeros((n_rows, end - start), dtype=np.float64)
                    for j in range(end - start):
                        nz = nzindices[i * self.batch_size + j]
                        x[nz, j] = gpu.retrieve_array(x_device[nz, j])
                else:
                    msg =  f"nzindices must be a list of non-zero indices for each column, got {type(nzindices)}"
                    LOGGER.error(msg)
                    raise TypeError(msg)

                # Append to output columns
                x_csc = sp.csc_matrix(x)
                x_csc.eliminate_zeros()
                out_cols.append(x_csc)

                # free memory
                if self.delete_rhs:
                    if end < n_rhs:
                        rhs = rhs[:, end:]  # overwrite rhs with remaining columns
                    else:
                        rhs = sp.csc_matrix((n_rows, 0))  # empty matrix

                progress.advance(task)
        except Exception as e:
            if isinstance(e, MemoryError):
                # remove objects
                progress.remove_task(task)
                try:
                    del rhs_batch
                except NameError:
                    pass
                try:
                    del rhs_batch_array
                except NameError:
                    pass
                try:
                    del rhs_device
                except NameError:
                    pass
                try:
                    del x
                except NameError:
                    pass
                try:
                    del x_device
                except NameError:
                    pass
                try:
                    del x_csc
                except NameError:
                    pass

                LOGGER.exception(
                    f"MemoryError during solving {e}. Falling back to LU solver for remaining columns."
                )
                if not self.delete_rhs:
                    remaining_rhs = rhs[:, start:]
                else:
                    remaining_rhs = rhs

                self.batch_size //= 2
                out_cols.extend(self.cholesky(remaining_rhs))

                return sp.hstack(out_cols).tocsc()
            else:
                progress.soft_stop()
                LOGGER.exception(f"Error during solving: {e}")
                raise e

        # Stack all columns into a sparse matrix
        out = sp.hstack(out_cols).tocsc()
        LOGGER.debug("Cholesky solving completed")
        progress.soft_stop()
        return out  # type: ignore

    def lu(self, rhs: sp.csc_matrix, nzindices: Optional[list] = None) -> sp.csc_matrix:
        """
        Solve the system using LU decomposition and single loop over rhs columns.

        This is efficient, as the factorized solver is used to solve each column of the right-hand side matrix
        independently. This is suitable for medium-sized matrices where the factorization is not too expensive.

        Note, this method instantiates an empty csc_matrix of the same shape as rhs and fills it with the solution
        for each column of rhs. This is is not as fast as saving the columns (possibly in batches) and stacking them
        at the end, but it is more memory efficient for large rhs matrices.
        """
        LOGGER.debug("Solving using LU decomposition")
        progress = PROGRESS.get_active_progress_bar(self.progress)
        n_rows, n_rhs = rhs.shape  # type: ignore
        out_cols = []
        num_batches = (n_rhs + self.batch_size - 1) // self.batch_size
        task = progress.add_task("LU solving batches", total=num_batches)
        try:
            for b in range(num_batches):
                # get current batch
                if self.delete_rhs:
                    n_rhs = rhs.shape[1]  # type: ignore
                    start = 0
                    end = min(self.batch_size, n_rhs)
                    rhs_batch = rhs[:, start:end]
                else:
                    start = b * self.batch_size
                    end = min((b + 1) * self.batch_size, n_rhs)
                    rhs_batch = rhs[:, start:end]

                # solve each column in the batch
                x_batch = []
                for i in range(rhs_batch.shape[1]):
                    nz = nzindices[b * self.batch_size + i] if nzindices is not None else np.arange(n_rows)
                    data = self.solver(rhs_batch[:, i].toarray().ravel())[nz]
                    col = sp.csc_matrix(
                        (data, (nz, np.zeros_like(nz))),
                        shape=(n_rows, 1),
                    )
                    x_batch.append(col)

                # save the result as a sparse matrix
                out_cols.append(sp.hstack(x_batch))

                # delete the processed columns from rhs to free memory
                if self.delete_rhs:
                    if end < n_rhs:
                        rhs = rhs[:, end:]  # overwrite rhs with remaining columns
                    else:
                        rhs = sp.csc_matrix((n_rows, 0))  # empty matrix
                progress.advance(task)
        except Exception as e:
            if isinstance(e, MemoryError):
                # remove objects
                progress.remove_task(task)
                try:
                    del rhs_batch
                except NameError:
                    pass
                try:
                    del x_batch
                except NameError:
                    pass

                LOGGER.exception(
                    f"MemoryError during solving {e}. Halving the batch size and retrying."
                )
                if not self.delete_rhs:
                    # NOTE: last batch failed, so the remaining rhs is from the start of that batch
                    remaining_rhs = rhs[:, start:]
                else:
                    remaining_rhs = rhs
                self.batch_size //= 2
                out_cols.extend(self.lu(remaining_rhs))

                return sp.hstack(out_cols).tocsc()
            else:
                progress.soft_stop()
                LOGGER.exception(f"Error during solving: {e}")
                raise e
        out = sp.hstack(out_cols).tocsc()
        LOGGER.debug("LU solving completed")
        progress.soft_stop()
        return out

    def lu_threaded(self, rhs: sp.csc_matrix) -> sp.csc_matrix:
        """
        Solve the system using LU decomposition on CPU, multithreaded over rhs columns.

        Args:
            rhs (sp.csc_matrix): The right-hand side matrix.

        Returns:
            sp.csc_matrix: The solved interior operator as a sparse matrix.
        """
        progress = PROGRESS.get_active_progress_bar(self.progress)
        LOGGER.debug("Solving using LU decomposition with multithreading")
        n_rhs = rhs.shape[1]  # type: ignore
        num_threads = self.NUM_CPU_THREADS
        chunk_size = (n_rhs + num_threads - 1) // num_threads

        task = progress.add_task("LU solving columns", total=n_rhs)

        def solve_chunk(col_range, advance_task=lambda: progress.advance(task)):
            results = []
            for i in col_range:
                x = self.solver(rhs[:, i].toarray().ravel())
                results.append((i, sp.csc_matrix(x.reshape(-1, 1))))
                advance_task()
            return results

        # Prepare column ranges for each thread
        col_ranges = [
            range(i, min(i + chunk_size, n_rhs)) for i in range(0, n_rhs, chunk_size)
        ]

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(solve_chunk, col_range) for col_range in col_ranges
            ]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())

        # Sort results by column index
        results.sort(key=lambda tup: tup[0])
        sorted_cols = [col for idx, col in results]

        # Stack all columns into a sparse matrix in the original order
        out = sp.hstack(sorted_cols).tocsc()
        LOGGER.debug("LU solving with multithreading completed")
        progress.soft_stop()

        return out  # type: ignore

    @property
    def batch_size(self) -> int:
        """
        Get the batch size for the solver.
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size: int) -> None:
        """
        Set the batch size for the solver.
        """
        if size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        self._batch_size = size
        LOGGER.debug(f"Batch size set to {self._batch_size}")
