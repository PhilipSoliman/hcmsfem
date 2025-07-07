from copy import copy
from datetime import datetime
from typing import Optional, Type

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp

from hcmsfem.boundary_conditions import (
    BoundaryCondition,
    BoundaryConditions,
    BoundaryType,
    HomogeneousDirichlet,
)
from hcmsfem.fespace import FESpace
from hcmsfem.logger import LOGGER, PROGRESS
from hcmsfem.meshes import BoundaryName, DefaultQuadMeshParams, MeshParams, TwoLevelMesh
from hcmsfem.operators import Operator
from hcmsfem.preconditioners import (
    CoarseSpace,
    OneLevelSchwarzPreconditioner,
    TwoLevelSchwarzPreconditioner,
)
from hcmsfem.problem_type import ProblemType
from hcmsfem.solvers import CustomCG


class Problem:
    def __init__(
        self,
        bcs: list[BoundaryConditions],
        mesh: TwoLevelMesh | MeshParams = DefaultQuadMeshParams.Nc4,
        ptype: ProblemType = ProblemType.CUSTOM,
        progress: Optional[PROGRESS] = None,
    ):
        """
        Initialize the problem with the given coefficient function, source function, and boundary conditions.

        Args:
            coefficient_function (callable): Function representing the coefficient in the PDE.
            source_function (callable): Function representing the source term in the PDE.
            boundary_conditions (dict): Dictionary containing boundary conditions for the problem.
        """
        self.progress = progress

        # get mesh
        if isinstance(mesh, MeshParams):
            try:
                self.two_mesh = TwoLevelMesh.load(
                    mesh,
                    progress=self.progress,
                )
            except FileNotFoundError:
                LOGGER.info("Mesh file not found. Creating a new mesh.")
                self.two_mesh = TwoLevelMesh(
                    mesh,
                    progress=self.progress,
                )
                self.two_mesh.save()
        elif isinstance(mesh, TwoLevelMesh):
            LOGGER.info("Using existing TwoLevelMesh.")
            self.two_mesh = mesh
        else:
            msg = (
                f"Invalid mesh type: {type(mesh)}. Expected TwoLevelMesh or MeshParams."
            )
            LOGGER.error(msg)
            raise TypeError(msg)
        
        # handle remaining input
        self.boundary_conditions = bcs
        self.ptype = ptype

        # set defaults
        self._linear_form = None
        self._linear_form_set = False
        self._bilinear_form = None
        self._bilinear_form_set = False

    def construct_fespace(
        self,
        fespaces: Optional[list] = None,
        orders: Optional[list] = None,
        dimensions: Optional[list] = None,
    ):
        """
        Construct the finite element space for the problem.

        Returns:
            FESpace: The finite element space for the problem.
        """
        if self.ptype == ProblemType.CUSTOM:
            if fespaces is None or orders is None or dimensions is None:
                LOGGER.error(
                    f"For custom problem type all of fespaces = {fespaces},"
                    f" orders = {orders}, and dimensions = {dimensions} must be provided."
                )
                raise ValueError(
                    "For custom problem type, fespaces, orders, and dimensions must be provided."
                )
            self.ptype.fespaces = fespaces
            self.ptype.orders = orders
            self.ptype.dimensions = dimensions
        self.fes = FESpace(
            self.two_mesh,
            self.boundary_conditions,
            ptype=self.ptype,
            progress=self.progress,
        )
        self._linear_form = ngs.LinearForm(self.fes.fespace)
        self._bilinear_form = ngs.BilinearForm(self.fes.fespace, symmetric=True)

    def get_trial_and_test_functions(self):
        """
        Get the trial and test functions for the finite element space.

        Returns:
            tuple: A tuple containing the trial and test functions.
        """
        if hasattr(self, "fes") is False:
            msg = "Finite element space has not been constructed yet. Call construct_fespace() first."
            LOGGER.error(msg)
            raise ValueError(msg)
        u = self.fes.u  # type: ignore
        v = self.fes.v  # type: ignore
        return u, v

    @property
    def linear_form(self) -> ngs.LinearForm:
        return self._linear_form

    @linear_form.setter
    def linear_form(self, lf: ngs.comp.SumOfIntegrals):
        if not isinstance(self._linear_form, ngs.LinearForm):
            msg = "Linear form has not been correctly initialized. Call construct_fespace() first."
            LOGGER.error(msg)
            raise ValueError(msg)
        if not isinstance(lf, ngs.comp.SumOfIntegrals):
            msg = "Linear form must be an instance of ngs.comp.SumOfIntegrals."
            LOGGER.error(msg)
            raise ValueError(msg)
        self._linear_form_set = True
        self._linear_form += lf

    @property
    def bilinear_form(self) -> ngs.BilinearForm:
        return self._bilinear_form

    @bilinear_form.setter
    def bilinear_form(self, bf: ngs.comp.SumOfIntegrals):
        if not isinstance(self._bilinear_form, ngs.BilinearForm):
            msg = "Bilinear form has not been correctly initialized. Call construct_fespace() first."
            LOGGER.error(msg)
            raise ValueError(msg)
        elif not isinstance(bf, ngs.comp.SumOfIntegrals):
            msg = "Bilinear form must be an instance of ngs.comp.SumOfIntegrals."
            LOGGER.error(msg)
            raise ValueError(msg)
        self._bilinear_form_set = True
        self._bilinear_form += bf

    def assemble(self, gfuncs: Optional[list[ngs.GridFunction]] = None):
        """
        Assemble the linear and bilinear forms.

        Assembles the system matrix based either on the provided linear and bilinear forms or problem type.

        Current supported problem types:
            - ProblemType.CUSTOM: Requires linear and bilinear forms to be set.
            - ProblemType.DIFFUSION: Requires two grid functions for coefficient and source terms, in that order.
        """
        LOGGER.info(
            f"Assembling system for {str(self.ptype)} and 1/H = {1/self.two_mesh.coarse_mesh_size:.0f}"
        )

        # check if linear and bilinear forms are instantiated on the finite element space
        if not isinstance(self._linear_form, ngs.LinearForm) or not isinstance(
            self._bilinear_form, ngs.BilinearForm
        ):
            msg = "Linear or bilinear form has not been initialized. Call construct_fespace() first."
            LOGGER.error(msg)
            raise ValueError(msg)

        # check problem type
        if self.ptype == ProblemType.CUSTOM:
            if not self._linear_form_set:
                msg = "Linear form has not been set."
                LOGGER.error(msg)
                raise ValueError(msg)
            if not self._bilinear_form_set:
                msg = "Bilinear form has not been set."
                LOGGER.error(msg)
                raise ValueError(msg)
        elif self.ptype == ProblemType.DIFFUSION:
            # check if gfuncs is provided for custom coefficient and source functions
            if gfuncs is None or len(gfuncs) != 2:
                msg = (
                    "For diffusion problem type, gfuncs must be provided with two grid functions."
                    "One for coefficient function and one for source function."
                )
                LOGGER.error(msg)
                raise ValueError(msg)

            # get trial and test functions
            u_h, v_h = self.get_trial_and_test_functions()

            # construct bilinear and linear forms
            self.bilinear_form = gfuncs[0] * ngs.grad(u_h) * ngs.grad(v_h) * ngs.dx
            self.linear_form = gfuncs[1] * v_h * ngs.dx
        elif self.ptype == ProblemType.NAVIER_STOKES:
            raise NotImplementedError(
                "Navier-Stokes problem type is not implemented yet."
            )
        else:
            raise ValueError(
                f"Problem type {self.ptype} is not supported. Use ProblemType.CUSTOM."
            )

        # solution grid function
        u = ngs.GridFunction(self.fes.fespace)

        # set boundary conditions on the grid function
        for bcs in self.boundary_conditions:
            bcs.set_boundary_conditions_on_gfunc(
                u, self.fes.fespace, self.two_mesh.fine_mesh
            )

        # assemble rhs and stiffness matrix
        b = self.linear_form.Assemble()
        A = self.bilinear_form.Assemble()

        return A, u, b

    def homogenize_rhs(
        self, A: ngs.Matrix, u: ngs.GridFunction, b: ngs.GridFunction
    ) -> ngs.Vector:
        res = b.vec.CreateVector()
        res.data = b.vec - A.mat * u.vec

        return res

    def restrict_system_to_free_dofs(
        self, A: ngs.Matrix, u: ngs.GridFunction, b: ngs.GridFunction
    ) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:

        res = self.homogenize_rhs(A, u, b)

        # free dofs
        free_dofs = self.fes.free_dofs_mask

        # export to numpy arrays & sparse matrix
        u_arr = copy(u.vec.FV().NumPy()[free_dofs])
        res_arr = res.FV().NumPy()[free_dofs]
        rows, cols, vals = A.mat.COO()
        A_sp = sp.csr_matrix((vals, (rows, cols)), shape=A.mat.shape)
        A_sp_f = A_sp[free_dofs, :][:, free_dofs]

        return A_sp_f, u_arr, res_arr

    def direct_ngs_solve(self, u: ngs.Vector, b: ngs.Vector, A: ngs.Matrix):
        """
        Solve the system of equations directly.

        Args:
            gfu (ngs.GridFunction): The grid function to store the solution.
            system (ngs.Matrix): The system matrix.
            load (ngs.Vector): The load vector.
        """
        # homogenization
        res = self.homogenize_rhs(A, u, b)

        # solve the system using sparse Cholesky factorization
        u.vec.data += (
            A.mat.Inverse(self.fes.fespace.FreeDofs(), inverse="sparsecholesky") * res
        )

    def solve(
        self,
        preconditioner: Optional[Type[Operator]] = None,
        coarse_space: Optional[Type[CoarseSpace]] = None,
        rtol: float = 1e-8,
        use_gpu: bool = False,
        save_cg_info: bool = False,
        save_coarse_bases: bool = False,
    ):
        system_str = f"Solving system for {str(self.ptype.name)} and 1/H = {1/self.two_mesh.coarse_mesh_size:.0f}"
        self.progress = PROGRESS.get_active_progress_bar(self.progress)
        task = self.progress.add_task(
            system_str, total=4 + save_cg_info + save_coarse_bases
        )
        LOGGER.info(system_str)

        # assemble the system
        A, self.u, b = self.assemble()
        self.progress.advance(task)

        # homogenization of the boundary conditions
        A_sp_f, u_arr, res_arr = self.restrict_system_to_free_dofs(A, self.u, b)
        self.progress.advance(task)

        # setup custom solver
        custom_cg = CustomCG(
            A_sp_f,
            res_arr,
            u_arr,
            tol=rtol,
            progress=self.progress,
        )

        # get preconditioner
        M_op = None
        precond = None
        coarse_space_bases = {}
        if preconditioner is not None:
            if isinstance(preconditioner, type):
                if preconditioner is OneLevelSchwarzPreconditioner:
                    precond = OneLevelSchwarzPreconditioner(
                        A_sp_f, self.fes, use_gpu=use_gpu, progress=self.progress
                    )
                elif preconditioner is TwoLevelSchwarzPreconditioner:
                    if coarse_space is None:
                        msg = "Coarse space must be provided for TwoLevelSchwarzPreconditioner."
                        LOGGER.error(msg)
                        raise ValueError(msg)
                    precond = TwoLevelSchwarzPreconditioner(
                        A_sp_f,
                        self.fes,
                        self.two_mesh,
                        coarse_space,
                        use_gpu=use_gpu,
                        progress=self.progress,
                    )
                    if save_coarse_bases:
                        coarse_space_bases = precond.get_restriction_operator_bases()
                else:
                    msg = f"Unknown preconditioner type: {preconditioner.__name__}"
                    LOGGER.error(msg)
                    raise ValueError(msg)
                if not use_gpu:
                    M_op = precond.as_linear_operator()
        self.precond_name = f"{precond}" if precond is not None else "None"
        self.progress.advance(task)

        success = False
        if not use_gpu:
            LOGGER.debug(
                f"Solving system on CPU:" f"\n\tpreconditioner: {self.precond_name}"
            )
            u_arr[:], success = custom_cg.sparse_solve(
                M_op, save_residuals=save_cg_info
            )
        else:
            LOGGER.debug(
                f"Solving system on GPU:" f"\n\tpreconditioner: {self.precond_name}"
            )
            u_arr[:], success = custom_cg.sparse_solve_gpu(precond, save_residuals=save_cg_info)  # type: ignore
        if not success:
            LOGGER.error(
                f"Conjugate gradient solver did not converge. Number of iterations: {custom_cg.niters}"
            )
        else:
            self.u.vec.FV().NumPy()[self.fes.free_dofs_mask] = u_arr
            LOGGER.debug("Saved solution to grid function.")
        self.progress.advance(task)

        # save cg coefficients if requested
        if save_cg_info:
            self.cg_alpha, self.cg_beta = custom_cg.alpha, custom_cg.beta
            self.cg_residuals = custom_cg.get_relative_residuals()
            self.cg_precond_residuals = None
            if precond is not None:
                self.cg_precond_residuals = (
                    custom_cg.get_relative_preconditioned_residuals()
                )
                LOGGER.debug("Obtained conjugate gradient preconditioned residuals.")
            self.approximate_eigs = custom_cg.get_approximate_eigenvalues_gpu()  # type: ignore
            LOGGER.debug("Obtained CG info")
            self.progress.advance(task)

        # save coarse operator grid functions if available
        if save_coarse_bases and coarse_space is not None:
            gfuncs = []
            names = []
            for basis, basis_gfunc in coarse_space_bases.items():
                names.append(basis)
                gfuncs.append(basis_gfunc)
            if gfuncs != []:
                self.save_ngs_functions(gfuncs, names, "coarse_bases")
                LOGGER.debug("Saved coarse space bases to file.")
            else:
                LOGGER.warning("No coarse space bases to save.")
            self.progress.advance(task)

        LOGGER.info("Solving system completed.")
        self.progress.soft_stop()

    def save_ngs_functions(
        self, funcs: list[ngs.GridFunction], names: list[str], category: str
    ):
        """
        Save the grid function solution to a file.

        Args:
            sol (ngs.GridFunction): The grid function containing the solution.
            name (str): The name of the problem (used for the filename).
        """
        # Get current date and time as a string
        LOGGER.info("Saving grid functions")
        now_str = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
        fn = f"{category}_{now_str}"
        tlm_dir = self.two_mesh.save_dir
        vtk = ngs.VTKOutput(
            self.two_mesh.fine_mesh,
            coefs=funcs,
            names=names,
            filename=str(tlm_dir / fn),
        )
        vtk.Do()
        LOGGER.info("Saved grid functions to %s", tlm_dir / fn)


if __name__ == "__main__":
    # get mesh parameters
    mesh_params = DefaultQuadMeshParams.Nc16

    # define boundary conditions
    bcs = HomogeneousDirichlet(ProblemType.DIFFUSION)
    bcs.set_boundary_condition(
        BoundaryCondition(
            name=BoundaryName.LEFT,
            btype=BoundaryType.DIRICHLET,
            values={0: 32 * ngs.y * (mesh_params.ly - ngs.y)},
        )
    )

    # construct finite element space
    problem = Problem([bcs], mesh=mesh_params)

    # construct finite element space
    problem.construct_fespace([ngs.H1], [1], [1])

    # get trial and test functions
    u_h, v_h = problem.get_trial_and_test_functions()

    # construct bilinear and linear forms
    problem.bilinear_form = ngs.grad(u_h) * ngs.grad(v_h) * ngs.dx
    problem.linear_form = (
        32 * (ngs.y * (mesh_params.ly - ngs.y) + ngs.x * (mesh_params.lx - ngs.x)) * v_h * ngs.dx
    )

    # assemble the forms
    A, u, b = problem.assemble()

    # direct solve
    problem.direct_ngs_solve(u, b, A)
    # problem.save_ngs_functions([u], ["solution"], "test_solution_direct")

    # cg solve
    problem.solve()
    # problem.save_ngs_functions([problem.u], ["solution"], "test_solution_cg")
