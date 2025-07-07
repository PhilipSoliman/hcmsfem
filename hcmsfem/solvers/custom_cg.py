from ctypes import CDLL, POINTER, byref, c_bool, c_double, c_int
from typing import Optional

import numpy as np
import torch
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse import diags as spdiags
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from tqdm import trange

import hcmsfem.solvers.clib.custom_cg as custom_cg_lib
from hcmsfem.eigenvalues import eigs
from hcmsfem.gpu_interface import GPUInterface
from hcmsfem.logger import LOGGER, PROGRESS
from hcmsfem.operators import Operator
from hcmsfem.root import get_root

# initialize GPU interface
gpu = GPUInterface()


class CustomCG:
    GPU_BATCH_SIZE = 128
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(
        self,
        A: np.ndarray | csr_matrix,
        b: np.ndarray,
        x_0: np.ndarray,
        tol: float = 1e-6,
        maxiter: Optional[int] = None,
        progress: Optional[PROGRESS] = None,
    ):
        self.progress = progress

        # system
        self.A = A
        self.b = b
        self.n = len(b)
        self.maxiter = maxiter if maxiter is not None else 10 * self.n

        # initial guess
        self.x_0 = x_0

        # initial residual
        self.r_0 = b - A @ x_0
        self.rho_0 = np.array([], dtype=np.float64)

        # convergence criteria
        self.tol = tol

        # residuals
        self.r_i = np.array([], dtype=np.float64)
        self.z_i = np.array([], dtype=np.float64)

        # exact solution
        self.x_exact = np.array([])
        self.exact_convergence = False
        self.e_i = np.array([], dtype=np.float64)

        # number of iterations required
        self.niters = 0

        # cg coefficients
        self.alpha = np.zeros([], dtype=np.float64)
        self.beta = np.zeros([], dtype=np.float64)

        # A eigen spectrum
        self.eigenvalues = np.array([], dtype=np.float64)
        self.eigenvectors = np.array([], dtype=np.float64)

        # polynomial coefficients
        self.residual_polynomials_coefficients = []

    def solve(
        self,
        save_residuals: bool = False,
        x_exact: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, bool]:
        alpha = np.zeros(self.maxiter, dtype=np.float64)
        beta = np.zeros(self.maxiter, dtype=np.float64)
        r_i = np.zeros(self.maxiter + 1, dtype=np.float64)
        x_m = np.zeros_like(self.x_0)
        e_i = np.zeros(self.maxiter + 1, dtype=np.float64)
        if x_exact.size > 0:
            self.exact_convergence = True

        niters = np.zeros(1, dtype=np.int32)
        success = custom_cg_lib.custom_cg(
            np.asarray(self.A),
            self.b,
            self.x_0,
            x_m,
            alpha,
            beta,
            self.n,
            niters,
            self.maxiter,
            self.tol,
            save_residuals,
            r_i,
            self.exact_convergence,
            x_exact,
            e_i,
        )

        # get number of iterations
        self.niters = niters[0]

        # save coefficients
        self.alpha = alpha[: self.niters]
        self.beta = beta[: (self.niters - 1)]

        # save residuals
        if save_residuals:
            self.r_i = r_i[: (self.niters + 1)]

        # save errors
        if self.exact_convergence:
            self.e_i = e_i[: (self.niters + 1)]

        return x_m, success

    def sparse_solve(
        self,
        M: Optional[LinearOperator] = None,
        save_residuals: bool = False,
    ) -> tuple[np.ndarray, int]:
        # complex dot product
        dotprod = np.vdot if np.iscomplexobj(self.x_0) else np.dot

        # matrix-vector product with A
        A = aslinearoperator(self.A)
        matvec = A.matvec

        # preconditioner matrix-vector product
        track_z = save_residuals and M is not None
        if M is None:
            M = LinearOperator(self.A.shape, lambda x: x)
        psolve = M.matvec

        # initial guess
        x = self.x_0.copy()

        # initial residual
        r = self.b - matvec(self.x_0) if self.x_0.any() else self.b.copy()
        rho_prev, p = None, None

        # intitial z vector
        z = psolve(r)

        # historic
        r_norm = np.linalg.norm(r)
        r_i = [r_norm]
        z_i = [np.linalg.norm(z)] if track_z else []
        alphas = []
        betas = []

        # main loop
        iteration = -1  # Ensure iteration is always defined
        success = False
        LOGGER.info("Starting CG iterations")
        self.progress = PROGRESS.get_active_progress_bar(self.progress)
        task = self.progress.add_task("CG iterations", total=None)
        desc = self.progress.get_description(task)
        desc += ": ({0:.0f}/{1:.0f})"
        desc += " [bold]| rel. r: {2:.2e} | α: {3:.2e}"
        desc += " | β: {4:.2e}[/bold]"
        for iteration in range(self.maxiter):
            if r_norm / r_i[0] < self.tol:  # relative residual tolerance
                success = True
                LOGGER.info(f"Converged after {iteration} iterations")
                break
            rho_cur = dotprod(r, z)
            if iteration > 0:
                beta = rho_cur / rho_prev  # type: ignore
                p *= beta  # type: ignore
                p += z  # type: ignore
                betas.append(beta)
            else:
                p = np.empty_like(r)
                p[:] = z[:]

            q = matvec(p)
            alpha = rho_cur / dotprod(p, q)
            x += alpha * p
            r -= alpha * q
            z = psolve(r)
            r_norm = np.linalg.norm(r)
            if save_residuals:
                r_i.append(r_norm)
            if track_z:
                z_i.append(np.linalg.norm(z))
            rho_prev = rho_cur
            alphas.append(alpha)

            # Update progres bar description
            self.progress.update(
                task,
                description=desc.format(
                    iteration + 1,
                    self.maxiter,
                    r_norm / r_i[0],
                    alpha,
                    betas[-1] if betas else 0,
                ),
            )

        if save_residuals:
            self.r_i = np.array(r_i)
        if track_z:
            self.z_i = np.array(z_i)

        self.alpha = np.array(alphas)
        self.beta = np.array(betas)
        self.niters = iteration

        self.progress.soft_stop()
        return x, success

    def sparse_solve_gpu(
        self,
        M: Operator | csc_matrix | None = None,
        save_residuals: bool = False,
    ) -> tuple[np.ndarray, int]:
        # dot product
        dotprod = torch.dot

        # matrix-vector product with A
        if isinstance(self.A, csc_matrix):
            A_device = gpu.send_matrix(self.A)
        else:
            A_device = gpu.send_matrix(csc_matrix(self.A))
        matvec = lambda x: torch.mv(A_device, x)

        # preconditioner matrix-vector product
        track_z = save_residuals and M is not None
        psolve = lambda x: x
        if isinstance(M, csc_matrix):
            M_device = gpu.send_matrix(M)
            psolve = lambda x: torch.mv(M_device, x)
        elif isinstance(M, Operator):
            psolve = lambda x: M.apply_gpu(x)

        # initial guess
        x = gpu.send_array(self.x_0.copy())

        # initial residual
        r = self.b - self.A @ self.x_0 if self.x_0.any() else self.b.copy()
        r = gpu.send_array(r)
        rho_prev, p = None, None

        # intitial z vector
        z = psolve(r)  # NOTE: tensor by consruction

        # historic
        r_i = [torch.norm(r)]
        z_i = [torch.norm(z)] if track_z else []
        alphas = []
        betas = []

        # main loop
        iteration = -1  # Ensure iteration is always defined
        success = False
        with trange(self.maxiter, desc="CG iterations", unit="it") as pbar:
            for iteration in range(self.maxiter):
                if r_i[-1] < self.tol:
                    success = True
                    pbar.n = (
                        iteration + 1
                    )  # Set progress bar to actual number of iterations
                    pbar.last_print_n = (
                        iteration + 1
                    )  # Force tqdm to print the final value
                    pbar.update(0)  # Refresh the bar
                    break
                rho_cur = dotprod(r, z)
                if iteration > 0:
                    beta = rho_cur / rho_prev  # type: ignore
                    p *= beta  # type: ignore
                    p += z  # type: ignore
                    betas.append(beta)
                else:
                    p = torch.empty_like(r)
                    p[:] = z[:]

                q = matvec(p)
                alpha = rho_cur / dotprod(p, q)
                x += alpha * p
                r -= alpha * q
                z = psolve(r)
                residual_norm = torch.norm(r).item()
                if save_residuals:
                    r_i.append(residual_norm)
                if track_z:
                    z_i.append(torch.norm(z))
                rho_prev = rho_cur
                alphas.append(alpha)

                # Update tqdm bar with current residual norm and alpha
                pbar.set_postfix(
                    {
                        "residual": f"{residual_norm:.2e}",
                        "alpha": f"{alpha:.2e}",
                        "beta": f"{betas[-1]:.2e}" if betas else "N/A",
                    }
                )
                pbar.update(1)

        if save_residuals:
            self.r_i = np.array(r_i)
        if track_z:
            self.z_i = gpu.retrieve_array(torch.stack(z_i))

        self.alpha = gpu.retrieve_array(torch.stack(alphas))
        self.beta = gpu.retrieve_array(torch.stack(betas))
        self.niters = iteration

        return gpu.retrieve_array(x), success

    def get_relative_errors(self) -> np.ndarray:
        if np.any(self.e_i):
            return self.e_i / self.e_i[0]
        else:
            raise ValueError("No errors saved")

    def get_relative_residuals(self) -> np.ndarray:
        if np.any(self.r_i):
            return self.r_i / self.r_i[0]
        else:
            raise ValueError("No residuals saved")

    def get_relative_preconditioned_residuals(self) -> np.ndarray:
        if np.any(self.z_i):
            return self.z_i / self.z_i[0]
        else:
            raise ValueError("No preconditioner residuals saved")

    def cg_polynomial(
        self,
        resolution: int,
        domain: tuple = (),
        eig_domain: tuple = (),
        respoly_error: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # calculate respoly coefficients
        if len(self.residual_polynomials_coefficients) == 0:
            self.residual_polynomials()

        # degree of the polynomial grows with every CG iteration and so does the number of coefficients
        degree = self.niters + 1

        # specify domain and resolution
        no_domain = domain == ()
        no_eig_domain = eig_domain == ()
        if no_eig_domain and no_domain:
            raise ValueError("Specify either eig_domain or domain.")

        if no_domain:
            eigenvalue_range = eig_domain[1] - eig_domain[0]
            domain_min = eig_domain[0] - 0.2 * eigenvalue_range
            domain_max = eig_domain[1] + 0.2 * eigenvalue_range
            domain = (domain_min, domain_max)

        # create x values
        x = np.linspace(domain[0], domain[1], resolution)

        # create residual polynomial & absolute error
        r = np.zeros((degree, resolution))
        for i in range(0, degree):
            coeffs = self.residual_polynomials_coefficients[i]
            rp = np.poly1d(coeffs)
            r[i] = rp(x)

        if respoly_error:
            e = self.calculate_respoly_error()
        else:
            e = np.empty(self.niters + 1)

        return x, r, e

    def get_approximate_eigenvalues(self, num_eigs: Optional[int] = None):
        """
        Returns the approximate eigenvalues of the system A using the Lanczos matrix.
        The eigenvalues are computed from the diagonal and off-diagonal elements of the Lanczos matrix.
        """
        LOGGER.debug("Calculating approximate eigenvalues using Lanczos matrix")
        if num_eigs is None:
            num_eigs = min(
                100, self.niters - 1
            )  # limit to 100 to save computation time
        eigenvalues = eigs(
            self.get_lanczos_matrix(), library="scipy", num_eigs=num_eigs, which="BE"
        )
        LOGGER.debug(f"Calculated {len(eigenvalues)} approximate eigenvalues")
        return eigenvalues

    def get_approximate_eigenvalues_gpu(self):
        LOGGER.debug("Calculating approximate eigenvalues using Lanczos matrix (GPU)")
        lanczos_matrix = self.get_lanczos_matrix()
        eigenvalues = eigs(lanczos_matrix)
        LOGGER.debug(f"Calculated {len(eigenvalues)} approximate eigenvalues (GPU)")
        return eigenvalues

    def get_lanczos_matrix(self) -> csc_matrix:
        delta = 1 / self.alpha + np.append(0, self.beta / self.alpha[:-1])
        eta = np.sqrt(self.beta) / self.alpha[:-1]
        return spdiags(
            [eta, delta, eta], offsets=[-1, 0, 1], shape=(self.niters, self.niters)  # type: ignore
        ).tocsc()

    def residual_polynomials(self) -> list[np.ndarray]:
        delta = 1 / self.alpha + np.append(0, self.beta / self.alpha[:-1])
        eta = np.append(0, np.sqrt(self.beta) / self.alpha[:-1])
        r_polynomials_coefficients = [[1], [1, -delta[0]]]
        rp_previous = np.poly1d(r_polynomials_coefficients[0])
        rp_current = np.poly1d(r_polynomials_coefficients[1])
        for i in range(2, self.niters + 1):
            # construct the next residual polynomial
            p = np.poly1d([1, -delta[i - 1]])
            p = np.polymul(p, rp_current)
            p = np.polyadd(p, -eta[i - 2] * rp_previous)
            p = p / eta[i - 1]
            r_polynomials_coefficients.append(list(p.coeffs))

            # update previous and current residual polynomials
            rp_previous = rp_current
            rp_current = p

        # normalize the coefficients
        r_polynomials_coefficients = [
            np.array(c) / c[-1] for c in r_polynomials_coefficients
        ]

        self.residual_polynomials_coefficients = r_polynomials_coefficients

        return r_polynomials_coefficients

    def calculate_respoly_error(self) -> np.ndarray:
        if self.residual_polynomials_coefficients == []:
            self.residual_polynomials()

        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.A)  # type: ignore

        # calculate initial residual in the eigenbasis of A
        self.rho_0 = self.eigenvectors.T @ self.r_0
        e = np.zeros(self.niters + 1)
        for i in range(self.niters + 1):
            coeffs = self.residual_polynomials_coefficients[i]
            rp = np.poly1d(coeffs)
            r_lambda = rp(self.eigenvalues)
            e[i] = np.sqrt(np.sum(r_lambda**2 * self.rho_0**2 / self.eigenvalues))

        return e

    # helper
    def calculate_iteration_upperbound(self) -> int:
        return CustomCG.calculate_iteration_upperbound_static(
            cond=np.linalg.cond(self.A),  # type: ignore
            log_rtol=np.log(self.tol),
            exact_convergence=self.exact_convergence,
        )

    @staticmethod
    def calculate_iteration_upperbound_static(
        cond: float, log_rtol: float, exact_convergence: bool = True
    ) -> int:
        """
        Assumes eigenvalues are uniformly distributed between lowest and highest eigenvalue. In this case, the
        classical CG convergence factor is given by f = (sqrt(cond) - 1) / (sqrt(cond) + 1), where cond is the condition
        number of A. The number of iterations required to reach a tolerance tol is given by ceil(log(tol / 2) / log(f)).
        """
        # convergence factor
        sqrt_cond = np.sqrt(cond)
        convergence_factor = (sqrt_cond - 1) / (sqrt_cond + 1)

        # convergence tolerance
        conv_tol = log_rtol - np.log(2)
        if (
            not exact_convergence
        ):  # See report Theorem: "Residual convergence criterion"
            conv_tol -= np.log(sqrt_cond)

        return int(np.ceil(conv_tol / np.log(convergence_factor)))

    def calculate_improved_cg_iteration_upperbound(
        self,
        clusters: list[tuple[float, float]],
    ) -> int:
        return CustomCG.calculate_improved_cg_iteration_upperbound_static(
            clusters=clusters,
            tol=self.tol,
            exact_convergence=self.exact_convergence,
        )

    @staticmethod
    def calculate_improved_cg_iteration_upperbound_static(
        clusters: list[tuple[float, float]],
        tol: float = 1e-6,
        exact_convergence: bool = True,
    ) -> int:
        """
        Calculates an improved CG iteration bound for non-uniform eigenspectra.
        Assumes available knowledge on the whereabouts of eigenvalue clusters
        """
        # setup
        log_rtol = np.log(tol)
        degrees = [0] * len(clusters)

        for i, cluster in enumerate(clusters):
            a_i, b_i = cluster
            log_rtol_eff = log_rtol
            for j in range(i):
                a_j, b_j = clusters[j]
                z_1 = (b_j + a_j - 2 * b_i) / (b_j - a_j)
                z_2 = (b_j + a_j) / (b_j - a_j)
                m_j = degrees[j]
                log_rtol_eff -= m_j * (
                    np.log(abs(z_1 - np.sqrt(z_1**2 - 1)) / (z_2 + np.sqrt(z_2**2 - 1)))
                )

            # calculate & store chebyshev degree
            degrees[i] = CustomCG.calculate_iteration_upperbound_static(
                cond=b_i / a_i,
                log_rtol=log_rtol_eff,
                exact_convergence=exact_convergence,
            )
        return sum(degrees)


if __name__ == "__main__":
    # test library access
    custom_cg_lib.TEST()
