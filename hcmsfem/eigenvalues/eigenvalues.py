from typing import Literal, Optional, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import torch.linalg as trl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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


def split_spectrum_into_clusters(
    spectrum: np.ndarray,
    n_clusters: Optional[int] = None,
    max_clusters: Optional[int] = 2,
    silhouette_score_th: Optional[float] = 0.9,
) -> list[tuple[float, float]]:
    """
    Returns the min and max of the array, or min/max of two clusters if n_clusters=2.
    Clustering is done on the log-scaled absolute values.

    Parameters:
        arr (np.ndarray): 1D array of values.
        n_clusters (optional, int): Number of clusters to form. If None, it will determine the optimal number of clusters using silhouette score.
        max_clusters (optional, int): Maximum number of clusters to consider if n_clusters is None.

    Returns:
        [tuple[float, float], ...]: List of tuples containing the min and max of each cluster.
    """
    # ensure spectrum is a numpy array
    spectrum = np.asarray(spectrum)

    # uniformly distributed eigenvalues or just end points given
    if n_clusters is not None and (n_clusters == 1 or len(spectrum) < 2):
        return [(np.min(spectrum), np.max(spectrum))]

    # Use square of log-scale for clustering
    X = np.log10(np.abs(spectrum)).reshape(-1, 1)**2

    if n_clusters is not None:
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(X)
    else:
        optimal_found = False
        for n_cluster in range(2, max_clusters + 1):
            # predict initial cluster centroids
            init = "k-means++"
            if n_cluster == 2:
                init = np.array([X.min(), X.max()]).reshape(-1, 1)
            elif n_cluster == 3:
                init = np.array(
                    [
                        X.min(),
                        np.mean(X),
                        X.max(),
                    ]
                ).reshape(-1, 1)
            labels = KMeans(
                n_clusters=n_cluster, init=init, n_init="auto", random_state=0
            ).fit_predict(X)
            score = silhouette_score(X, labels)
            if score > silhouette_score_th:
                optimal_found = True
                n_clusters = n_cluster
                LOGGER.debug(
                    f"Optimal number of clusters determined: {n_clusters} (score: {score:.4f})"
                )
                break
        if not optimal_found:
            LOGGER.debug(
                f"Could not find optimal number of clusters with confidence {silhouette_score_th}. Returning single cluster."
            )
            return [(np.min(spectrum), np.max(spectrum))]


    clusters = []
    for i in range(n_clusters):
        cluster_vals = spectrum[labels == i]
        clusters.append(
            (np.min(cluster_vals), np.max(cluster_vals), np.mean(cluster_vals))
        )

    # Sort clusters by mean value
    clusters = sorted(clusters, key=lambda x: x[2])

    return [(float(c[0]), float(c[1])) for c in clusters]
