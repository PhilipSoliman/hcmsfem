from typing import Literal, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import torch.linalg as trl

from hcmsfem.gpu_interface import GPUInterface
from hcmsfem.logger import LOGGER

# initialize gpu
gpu = GPUInterface()


def eigs(
    A: sp.csc_matrix | sp.csr_matrix,
    library: Union[Literal["torch"], Literal["scipy"]] = "torch",
    num_eigs: int = 100,
    which: str = "BE",
) -> np.ndarray:
    """
    Compute the approximate eigenvalues of the sparse matrix A.

    Args:
        library (str): The library to use for computing eigenvalues ('torch' or 'scipy').
        num_eigs (int): The number of eigenvalues to compute.
        which (str): Specifies which eigenvalues to compute ('BE' for both ends).

    Returns:
        np.ndarray: The computed eigenvalues.
    """
    if library == "torch":
        if gpu.AVAILABLE:
            LOGGER.debug("Using GPU for eigenvalue computation")
            return _torch_eigs(A)
        else:
            LOGGER.error(gpu.AVAILABILTY_MSG)
            raise RuntimeError(gpu.AVAILABILTY_MSG)
    elif library == "scipy":
        LOGGER.debug("Using SciPy for eigenvalue computation")
        return _scipy_eigs(A, num_eigs, which)
    else:
        raise ValueError("Unsupported library. Use 'torch' or 'scipy'.")


def _scipy_eigs(
    A: sp.csc_matrix | sp.csr_matrix, num_eigs: int, which: str
) -> np.ndarray:
    """
    Returns the approximate eigenvalues of the system A using the Lanczos matrix.
    The eigenvalues are computed from the diagonal and off-diagonal elements of the Lanczos matrix.
    """
    eigenvalues = spl.eigsh(
        A,
        k=num_eigs,
        which=which,  # gets eigenvalues on both ends of the spectrum
        return_eigenvectors=False,
    )
    LOGGER.debug(f"Calculated {len(eigenvalues)} approximate eigenvalues")
    return eigenvalues


def _torch_eigs(A: sp.csc_matrix | sp.csr_matrix) -> np.ndarray:
    A_gpu = gpu.send_matrix(A, dense=True)
    eigenvalues = gpu.retrieve_array(trl.eigvalsh(A_gpu))
    return eigenvalues
