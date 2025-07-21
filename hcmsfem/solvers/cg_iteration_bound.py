from enum import Enum
from typing import Optional

import numpy as np
from sympy import Lambda

from hcmsfem.logger import LOGGER


def classic_cg_iteration_bound(
    k: float, log_rtol: float = np.log(1e-6), exact_convergence: bool = True
) -> int:
    """
    Assumes eigenvalues are uniformly distributed between lowest and highest eigenvalue. In this case, the
    classical CG convergence factor is given by f = (sqrt(cond) - 1) / (sqrt(cond) + 1), where cond is the condition
    number of A. The number of iterations required to reach a tolerance log_rtol is given by ceil(log(log_rtol / 2) / log(f)).

    Args:
        k (float): The condition number of the system.
        log_rtol (float): The log of the convergence tolerance. Defaults to 1e-6.
        exact_convergence (bool): If True, log_rtol is the log of the relative error tolerance convergence criterion,
            otherwise log_rtol corresponds to the log of the relative residual tolerance. Defaults to True.
    """
    if k > 1:
        # convergence factor
        sqrt_cond = np.sqrt(k)
        convergence_factor = (sqrt_cond - 1) / (sqrt_cond + 1)

        # convergence tolerance
        conv_tol = log_rtol - np.log(2)
        if (
            not exact_convergence
        ):  # See report Theorem: "Residual convergence criterion"
            conv_tol -= np.log(sqrt_cond)

        return int(np.ceil(conv_tol / np.log(convergence_factor)))
    elif k == 1:
        # if condition number is 1, then the system only requires one iteration
        return 1


def split_eigenspectrum(eigs: np.ndarray) -> int:
    """
    Splits eigenspectrum between the two eigenvalues that have the largest mutual logarithmic distance.

    Args:
        eigs (list or np.ndarray): The sorted eigenvalues of the system.
    Returns:
        int: The index of the eigenvalue after which the spectrum should be split.
    """
    eigs = np.asarray(eigs)
    log_distances = np.abs(np.log(eigs[1:] / eigs[:-1]))
    return np.argmax(log_distances)


def condition_number_threshold(k: float, k_l: float, k_r: float) -> bool:
    """
    Checks if the condition number of the partitioned system is above a threshold.

    Args:
        k (float): The condition number of the original system.
        k_l (float): The condition number of the left partition.
        k_r (float): The condition number of the right partition.
    Returns:
        bool: True if the condition number is above the threshold, False otherwise.
    """
    x = -1 / (4 * np.sqrt(k_r) * np.exp(1 / (2 * np.sqrt(k_r))))
    L = np.log(-x)
    l = np.log(-L)
    threshold = 4 * k_l * k_r * (L - l + l / L) ** 2
    return k > threshold


def partition_eigenspectrum(eigs: np.ndarray) -> list[int]:
    """
    Recursively splits the eigenspectrum into clusters based on the condition number threshold.

    Args:
        eigs (np.ndarray): The sorted eigenvalues of the system.
    Returns:
        list[int]: The partition indices.
    """
    if len(eigs) == 0:
        return []
    elif len(eigs) <= 2:
        # always return the index of the second eigenvalue
        # this ensures that we have at least one cluster
        return [len(eigs) - 1]
    k = eigs[-1] / eigs[0]
    split_index = split_eigenspectrum(eigs)
    k_l = eigs[split_index] / eigs[0]
    k_r = eigs[-1] / eigs[split_index + 1]

    if condition_number_threshold(k, k_l, k_r):
        return np.concatenate(
            (
                partition_eigenspectrum(eigs[: split_index + 1]),
                split_index
                + 1
                + np.array(partition_eigenspectrum(eigs[split_index + 1 :])),
            )
        ).tolist()
    else:
        return [len(eigs) - 1]


def multi_cluster_cg_iteration_bound(
    clusters: list[tuple[float, float]],
    log_rtol: float = np.log(1e-6),
    exact_convergence: bool = True,
) -> int:
    """
    Calculates an improved CG iteration bound for non-uniform eigenspectra.
    Assumes available knowledge on the whereabouts of eigenvalue clusters

    Args:
        clusters (list[tuple[float, float]]): List of tuples representing the clusters of eigenvalues.
            Each tuple contains the lower and upper bounds of the cluster.
        log_rtol (float): The log of the convergence tolerance. Defaults to np.log(1e-6).
        exact_convergence (bool): If True, log_rtol is the log of the relative error tolerance convergence criterion,
            otherwise log_rtol corresponds to the log of the relative residual tolerance. Defaults to True.
    """
    degrees = [0] * len(clusters)

    # find all clusters with unit condition number
    unit_clusters = [i for i, (a, b) in enumerate(clusters) if a == b]

    for i, cluster in enumerate(clusters):
        if i in unit_clusters:
            continue
        a_i, b_i = cluster

        # subtract unit cluster contributions from log_rtol
        log_rtol_eff = log_rtol - np.sum(
            [np.log(abs(1 - b_i / clusters[u][0])) for u in unit_clusters]
        )
        for j in range(i):
            a_j, b_j = clusters[j]
            if a_j != b_j:
                z_1 = (b_j + a_j - 2 * b_i) / (b_j - a_j)
                z_2 = (b_j + a_j) / (b_j - a_j)
                m_j = degrees[j]
                log_rtol_eff -= m_j * (
                    np.log(abs(z_1 - np.sqrt(z_1**2 - 1)) / (z_2 + np.sqrt(z_2**2 - 1)))
                )
            # else:
            #     # if the cluster has unit condition number, we assume a regular polynomial
            #     log_rtol_eff -= np.log(abs(1 - b_i / a_j))

        # calculate & store chebyshev degree
        degrees[i] = classic_cg_iteration_bound(
            b_i / a_i,
            log_rtol=log_rtol_eff,
            exact_convergence=exact_convergence,
        )
    return np.sum(degrees) + len(unit_clusters)  # add the unit clusters count


def sharpened_cg_iteration_bound(
    eigs: np.ndarray, log_rtol: float = np.log(1e-6), exact_convergence: bool = True
) -> int:
    """
    Calculates a sharpened CG iteration bound for non-uniform eigenspectra.

    Args:
        eigs (np.ndarray): The sorted (approximate) eigenvalues of the system.
        log_rtol (float): The log of the convergence tolerance. Defaults to np.log(1e-6).
        exact_convergence (bool): If True, log_rtol is the log of the relative error tolerance convergence criterion,
            otherwise log_rtol corresponds to the log of the relative residual tolerance. Defaults to True.
    """
    eigs = np.asarray(eigs)
    if len(eigs) == 1:
        # Only one eigenvalue provided, not enough information to calculate bound
        return 1
    if not np.all(eigs[:-1] <= eigs[1:]):
        eigs = np.sort(eigs)  # Ensure eigenvalues are sorted in ascending order
    if eigs[0] <= 0:
        raise ValueError("All eigenvalues must be positive.")
    partition_indices = partition_eigenspectrum(eigs)
    clusters = []
    start = 0
    for end in partition_indices:
        clusters.append((eigs[start], eigs[end]))
        start = end + 1
    return multi_cluster_cg_iteration_bound(
        clusters, log_rtol=log_rtol, exact_convergence=exact_convergence
    )


def tail_condition(p: int, k: float, k_l: float, k_r: float, log_rtol: float) -> bool:
    if k_l > k_r:
        return False  # expansion is not valid

    sparsity_condition = p < np.sqrt(k_l) / 2 * (np.log(2) - log_rtol)
    uniform_performance_threshold = p < np.sqrt(k / k_r) * (
        np.log(2) - log_rtol
    ) / np.log(4 * k / k_l)
    two_cluster_performance_threshold = (
        p
        < np.sqrt(k_l)
        * (np.log(2) - log_rtol)
        * (1 / (np.sqrt(k_r) * np.log(4 * k / k_l)) + 1 / 2)
        + 1
    )
    return (
        sparsity_condition
        and uniform_performance_threshold
        and two_cluster_performance_threshold
    )


def partition_mixed_eigenspectrum(
    eigs: np.ndarray,
    log_rtol: float,
    tail_eigenvalues: list[float] = [],
    tail_indices: list[int] = [],
    start_idx: int = 0,
) -> list[int]:
    """
    Recursively splits the eigenspectrum into tail eigenvalues and clusters based on the sparsity condition. If the sparsity condition is not met, it clusters based on the condition number threshold.

    Args:
        eigs (np.ndarray): The sorted eigenvalues of the system.
        log_rtol (float): The log of the convergence tolerance.
        tail_eigenvalues (list[float]): empty list to hold tail eigenvalues.
        tail_indices (list[int]): empty list to hold starting indices of tail cluster.
    Returns:
        list[int]: The partition indices.
    """
    if len(eigs) == 0:
        return []
    if len(eigs) == 1:  # add single eigenvalue clusters to tail by default
        tail_eigenvalues.append(eigs[0])
        tail_indices.append(start_idx)
        return [0]
    k = eigs[-1] / eigs[0]
    split_index = split_eigenspectrum(eigs)
    k_l = eigs[split_index] / eigs[0]
    k_r = eigs[-1] / eigs[split_index + 1]
    p = split_index + 1

    # check sparsity condition
    if tail_condition(p, k, k_l, k_r, log_rtol=log_rtol):
        tail_eigenvalues += eigs[:p].tolist()
        tail_indices.append(start_idx)
        return [split_index] + (
            p
            + np.array(
                partition_mixed_eigenspectrum(
                    eigs[p:],
                    log_rtol=log_rtol,
                    tail_eigenvalues=tail_eigenvalues,
                    tail_indices=tail_indices,
                    start_idx=start_idx + p,
                )
            )
        ).tolist()
    elif condition_number_threshold(k, k_l, k_r):
        return (
            partition_mixed_eigenspectrum(
                eigs[:p],
                log_rtol=log_rtol,
                tail_eigenvalues=tail_eigenvalues,
                tail_indices=tail_indices,
                start_idx=start_idx,
            )
            + (
                p
                + np.array(
                    partition_mixed_eigenspectrum(
                        eigs[p:],
                        log_rtol=log_rtol,
                        tail_eigenvalues=tail_eigenvalues,
                        tail_indices=tail_indices,
                        start_idx=start_idx + p,
                    )
                )
            ).tolist()
        )
    else:
        return [len(eigs) - 1]


def mixed_multi_cluster_cg_iteration_bound(
    clusters: list[tuple[float, float]],
    tail_eigenvalues: list[float],
    log_rtol: float = np.log(1e-6),
    exact_convergence: bool = True,
) -> int:
    """
    Similar to `multi_cluster_cg_iteration_bound`, but allows for a list of tail eigenvalues to be passed in.

    Args:
        clusters (list[tuple[float, float]]): List of tuples representing the clusters of eigenvalues.
            Each tuple contains the lower and upper bounds of the cluster.
        tail_eigenvalues (list[float]): List of tail eigenvalues.
        log_rtol (float): The log of the convergence tolerance. Defaults to np.log(1e-6).
        exact_convergence (bool): If True, log_rtol is the log of the relative error tolerance convergence criterion,
            otherwise log_rtol corresponds to the log of the relative residual tolerance. Defaults to True.
    """
    degrees = [0] * len(clusters)

    # Each factor is (1 - x / lambda_i) = (-1/lambda_i)x + 1
    factors = [np.poly1d([-1.0 / eig, 1.0]) for eig in tail_eigenvalues]

    # Multiply all factors together
    r_poly = np.poly1d([1.0])  # start with the constant polynomial 1
    for f in factors:
        r_poly *= f

    for i, cluster in enumerate(clusters):
        a_i, b_i = cluster
        log_rtol_eff = log_rtol - np.sum(
            [np.log(abs(1 - b_i / eig)) for eig in tail_eigenvalues]
        )
        for j in range(i):
            a_j, b_j = clusters[j]
            z_1 = (b_j + a_j - 2 * b_i) / (b_j - a_j)
            z_2 = (b_j + a_j) / (b_j - a_j)
            m_j = degrees[j]
            log_rtol_eff -= m_j * (
                np.log(abs(z_1 - np.sqrt(z_1**2 - 1)) / (z_2 + np.sqrt(z_2**2 - 1)))
            )

        # calculate & store chebyshev degree
        degrees[i] = classic_cg_iteration_bound(
            b_i / a_i,
            log_rtol=log_rtol_eff,
            exact_convergence=exact_convergence,
        )
    return np.sum(degrees) + len(tail_eigenvalues)  # add the tail eigenvalues count


def mixed_sharpened_cg_iteration_bound(
    eigs: np.ndarray, log_rtol: float = np.log(1e-6), exact_convergence: bool = True
) -> int:
    """
    Calculates a sharpened CG iteration bound for non-uniform eigenspectra.

    Args:
        eigs (np.ndarray): The sorted (approximate) eigenvalues of the system.
        log_rtol (float): The log of the convergence tolerance. Defaults to np.log(1e-6).
        exact_convergence (bool): If True, log_rtol is the log of the relative error tolerance convergence criterion,
            otherwise log_rtol corresponds to the log of the relative residual tolerance. Defaults to True.
    """
    eigs = np.asarray(eigs)
    if len(eigs) == 1:
        # Only one eigenvalue provided, not enough information to calculate bound
        return 1
    if not np.all(eigs[:-1] <= eigs[1:]):
        eigs = np.sort(eigs)  # Ensure eigenvalues are sorted in ascending order
    if eigs[0] <= 0:
        raise ValueError("All eigenvalues must be positive.")
    tail_eigenvalues = []
    tail_indices = []
    partition_indices = partition_mixed_eigenspectrum(
        eigs,
        log_rtol=log_rtol,
        tail_eigenvalues=tail_eigenvalues,
        tail_indices=tail_indices,
    )
    clusters = []
    start = 0
    for end in partition_indices:
        if not np.isin(start, tail_indices):  # skip over tail clusters
            clusters.append((eigs[start], eigs[end]))
        start = end + 1
    return mixed_multi_cluster_cg_iteration_bound(
        clusters,
        tail_eigenvalues=tail_eigenvalues,
        log_rtol=log_rtol,
        exact_convergence=exact_convergence,
    )


class CGIterationBound:
    """
    Attributes
    ----------
        classic (tuple[int, int]): The CG iteration and the classic CG iteration bound calculated from the spectrum at that iteration. Returns the latest bound.
        multi_cluster (tuple[int, int]): The sharpened multi-cluster CG iteration bound, same format as `classic`.
        tail_cluster (tuple[int, int]): The sharpened multi-cluster-tail CG iteration bound, same format as `classic`.
        estimate (tuple[int, int]): An estimate of CG's iteration count at convergence, same format as `classic`.
        classic_l (list[tuple[int, int]]): List of tuples containing the iteration and the classic CG iteration bound.
        multi_cluster_l (list[tuple[int, int]]): List of tuples containing the iteration and the multi-cluster CG iteration bound.
        tail_cluster_l (list[tuple[int, int]]): List of tuples containing the iteration and the tail cluster CG iteration bound.
        estimate_l (list[tuple[int, int]]): List of tuples containing the iteration and the estimate of CG's iteration count at convergence.

    How to use
    ----------
    ```python
        eps = 1e-8 # CG relative error tolerance
        cg_iter_bound = CGIterationBound(log_rtol=(np.log(epsilon)))
        i = 0 # CG iteration index
        update_frequency = 5 # update every 5 iterations
        while ... #CG has not converged:
            # CG iteration here
            # ...
            i += 1

            if i % update_frequency == 0:
                # get Lanczos coefficients from CG coefficients, alpha & beta
                # calculate approximate spectrum from Lanczos matrix (Ritz values)
                cg_iter_bound.update(spectrum)
        cg_iter_bound.show()
    ```

    """

    _CLASSIC_BOUND_CLUSTER_THRESHOLD: int = 1
    _MULTI_CLUSTER_BOUND_CLUSTER_THRESHOLD: int = 2
    _CLUSTER_THRESHOLD_NOT_MET = "Need to have detected at least {0:.0f} clusters to access this bound. Currently detected {1} cluster(s)."
    _FORMAT_STRING_WIDTH = 14

    def __init__(self, log_rtol: float = np.log(1e-6), exact_convergence: bool = True):
        self.log_rtol = log_rtol
        self.exact_convergence = exact_convergence

        # bounds
        self.classic_l = []  # lists of tuples (iteration, bound)
        self.multi_cluster_l = []
        self.tail_cluster_l = []

        # iteration estimate
        self.estimate_l = []  # lists of tuples (iteration, estimate)

        # bookkeeping
        self._iteration = 0
        self._num_clusters_detected = 0
        self._num_converged_spectra = 0
        self._previous_clusters = np.array([])
        self._converged_clusters = np.array([])

    def update(self, spectrum: np.ndarray):
        """
        Updates the CG iteration bound with a new (approximate) spectrum.
        Args:
            spectrum (np.ndarray): The current spectrum of eigenvalues.
        """

        # CG iteration index
        iteration = len(spectrum)

        # partition the current eigenvalues (into clusters)
        partition_indices = partition_eigenspectrum(spectrum)

        # form clusters from the partition indices
        current_clusters = np.zeros(len(partition_indices) + 1, dtype=float)
        start = 0
        for i, end in enumerate(partition_indices):
            current_clusters[i] = spectrum[start]
            current_clusters[i + 1] = spectrum[end]
            start = end + 1

        # check for cluster convergence
        all_close = False
        if len(self._previous_clusters) == len(current_clusters):
            all_close = np.allclose(
                self._previous_clusters / current_clusters, 1, atol=1e-1
            )

        # check for earlier cluster convergence
        found_earlier_convergence = False
        if len(self._converged_clusters) == len(current_clusters):
            found_earlier_convergence = np.allclose(
                self._converged_clusters / current_clusters, 1, atol=1e-1
            )

        if all_close and not found_earlier_convergence:
            # save the current clusters as converged
            self._converged_clusters = current_clusters

            # update the number of clusters detected
            self._num_clusters_detected = len(partition_indices)

            # increment the number of converged spectra
            self._num_converged_spectra += 1

            # store spectrum
            self.spectrum = spectrum

            # update the bounds
            self._update_bounds(iteration)
            LOGGER.info(
                f"Novel cluster convergence detected at iteration {iteration}. Bounds updated.",
            )

        # update the previous cluster bounds
        self._previous_clusters = current_clusters

    @property
    def classic(self) -> Optional[int]:
        if self._num_clusters_detected > self._CLASSIC_BOUND_CLUSTER_THRESHOLD:
            LOGGER.debug(
                self._CLUSTER_THRESHOLD_NOT_MET.format(
                    self._CLASSIC_BOUND_CLUSTER_THRESHOLD,
                    self._num_clusters_detected,
                )
            )
        return self._classic[-1] if self.classic_l else None

    @property
    def multi_cluster(self) -> Optional[tuple[int, int]]:
        if self._num_clusters_detected > self._MULTI_CLUSTER_BOUND_CLUSTER_THRESHOLD:
            LOGGER.debug(
                self._CLUSTER_THRESHOLD_NOT_MET.format(
                    self._MULTI_CLUSTER_BOUND_CLUSTER_THRESHOLD,
                    self._num_clusters_detected,
                )
            )
        return self.multi_cluster_l[-1] if self.multi_cluster_l else None

    @property
    def tail_cluster(self) -> Optional[tuple[int, int]]:
        return self.tail_cluster_l[-1] if self.tail_cluster_l else None

    @property
    def estimate(self) -> Optional[tuple[int, int]]:
        return self._estimate_l[-1] if self._estimate_l else None

    def show(self):
        with LOGGER.setLevelContext(LOGGER.INFO):  # ensure string is visible in console
            LOGGER.info(str(self))

    # helper
    def _update_bounds(self, iteration: int) -> None:
        if self._num_clusters_detected >= self._CLASSIC_BOUND_CLUSTER_THRESHOLD:
            k = self.spectrum[-1] / self.spectrum[0]
            self.classic_l.append(
                (
                    iteration,
                    classic_cg_iteration_bound(
                        k,
                        log_rtol=self.log_rtol,
                        exact_convergence=self.exact_convergence,
                    ),
                )
            )
        if self._num_clusters_detected >= self._MULTI_CLUSTER_BOUND_CLUSTER_THRESHOLD:
            self.multi_cluster_l.append(
                (
                    iteration,
                    sharpened_cg_iteration_bound(
                        self.spectrum,
                        log_rtol=self.log_rtol,
                        exact_convergence=self.exact_convergence,
                    ),
                )
            )

        # NOTE: can always calculate the tail cluster bound
        self.tail_cluster_l.append(
            (
                iteration,
                mixed_sharpened_cg_iteration_bound(
                    self.spectrum,
                    log_rtol=self.log_rtol,
                    exact_convergence=self.exact_convergence,
                ),
            )
        )

        if self.multi_cluster is not None:
            self.estimate_l.append(
                (
                    iteration,
                    int(np.ceil((self.tail_cluster[1] + self.multi_cluster[1]) / 2)),
                )
            )

    # output string
    def __str__(self):
        w = self._FORMAT_STRING_WIDTH
        tolerance_type = (
            "relative error" if self.exact_convergence else "relative residual"
        )
        tolerance_str = f"{tolerance_type} < {np.exp(self.log_rtol):.2e}"

        pre_str = f"\t"
        classic_str = ""
        if self.classic_l:
            for i, bound in self.classic_l:
                classic_str += f"\n\t{pre_str}{'i=' + str(i) + '->' + str(bound)}"
        else:
            classic_str = f"\n\t{pre_str}{None}"

        multi_cluster_str = ""
        if self.multi_cluster_l:
            for i, bound in self.multi_cluster_l:
                multi_cluster_str += (
                    f"\n\t{pre_str}{'i=' + str(i) + '->' + str(bound)}"
                )
        else:
            multi_cluster_str = f"\n\t{pre_str}{None}"

        tail_cluster_str = ""
        if self.tail_cluster_l:
            for i, bound in self.tail_cluster_l:
                tail_cluster_str += (
                    f"\n\t{pre_str}{'i=' + str(i) + '->' + str(bound)}"
                )
        else:
            tail_cluster_str = f"\n\t{pre_str}{None}"

        estimate_str = ""
        if self.estimate_l:
            for i, estimate in self.estimate_l:
                estimate_str += f"\n\t{pre_str}{'i=' + str(i) + '->' + str(estimate)}"
        else:
            estimate_str = f"\n\t{pre_str}{None}"
        return (
            "[bold]CG Iteration Bounds[/bold]"
            f"\n\t{'classic':<{w}}{classic_str}"
            f"\n\t{'multi-cluster':<{w}}{multi_cluster_str}"
            f"\n\t{'tail-cluster':<{w}}{tail_cluster_str}"
            f"\n\n[bold]CG Iteration Estimate[/bold]"
            f"\n\t{'estimate':<{w}}{estimate_str}"
            f"\n\n[bold]Convergence criterion[/bold]"
            f"\n\t{tolerance_str}"
            f"\n\n[bold]Spectrum Information[/bold]"
            f"\n\tspectrum size (most recent): {len(self.spectrum)}"
            f"\n\tnumber of clusters detected: {self._num_clusters_detected}"
            f"\n\tnumber of converged spectra: {self._num_converged_spectra}"
        )