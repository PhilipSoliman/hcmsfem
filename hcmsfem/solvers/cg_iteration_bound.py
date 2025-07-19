import numpy as np
from sympy import Lambda


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
        return (
            1  # Only one eigenvalue provided, not enough information to calculate bound
        )
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
