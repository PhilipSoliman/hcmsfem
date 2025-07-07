from enum import Enum
from typing import Callable, Optional

import ngsolve as ngs
import numpy as np

from hcmsfem.boundary_conditions import BoundaryConditions, HomogeneousDirichlet
from hcmsfem.logger import LOGGER, PROGRESS
from hcmsfem.meshes import DefaultQuadMeshParams, MeshParams, TwoLevelMesh
from hcmsfem.preconditioners import (
    AMSCoarseSpace,
    GDSWCoarseSpace,
    OneLevelSchwarzPreconditioner,
    RGDSWCoarseSpace,
    TwoLevelSchwarzPreconditioner,
)
from hcmsfem.problem_type import ProblemType
from hcmsfem.problems.problem import Problem


class SourceFunc(Enum):
    """Enumeration for source functions used in the diffusion problem."""

    CONSTANT = ("constant_source", r"$\mathcal{S}_{\text{const}}$", "const")
    PARABOLIC = ("parabolic_source", r"$\mathcal{S}_{\text{par}}$", "par")

    def __init__(self, func_name: str, latex: str, short_name: str):
        self._func_name = func_name
        self._latex = latex
        self._short_name = short_name

    def get_func(self, obj: "DiffusionProblem") -> Callable:
        """Return the function name associated with the coefficient function."""
        return getattr(obj, self._func_name)

    @property
    def latex(self):
        """Return the LaTeX representation associated with the coefficient function."""
        return self._latex

    @property
    def short_name(self):
        """Return the short name associated with the coefficient function."""
        return self._short_name


class CoefFunc(Enum):
    """Enumeration for coefficient functions used in the diffusion problem."""

    SINUSOIDAL = ("sinusoidal_coefficient", r"$\mathcal{C}_{\mathrm{sin}}$", "sin")
    VERTEX_INCLUSIONS = (
        "vertex_centered_inclusions_coefficient",
        r"$\mathcal{C}_{\mathrm{vert}}$",
        "vert",
    )
    TWO_LAYER_VERTEX_INCLUSIONS = (
        "two_layer_vertex_centered_inclusions_coefficient",
        r"$\mathcal{C}_{\mathrm{2layer, \ vert}}$",
        "2lvert",
    )
    THREE_LAYER_VERTEX_INCLUSIONS = (
        "three_layer_vertex_centered_inclusions_coefficient",
        r"$\mathcal{C}_{\mathrm{3layer, \ vert}}$",
        "3lvert",
    )
    EDGE_CENTERED_INCLUSIONS = (
        "edge_centered_inclusions_coefficient",
        r"$\mathcal{C}_{\mathrm{edge}}$",
        "edge",
    )
    SINGLE_SLAB_EDGE_INCLUSIONS = (
        "single_slab_edge_inclusions_coefficient",
        r"$\mathcal{C}_{\mathrm{single \ slab, \ edge}}$",
        "sslab_edge",
    )
    DOUBLE_SLAB_EDGE_INCLUSIONS = (
        "double_slab_edge_inclusions_coefficient",
        r"$\mathcal{C}_{\mathrm{double \ slab, \ edge}}$",
        "dslab_edge",
    )
    EDGE_SLABS_AROUND_VERTICES_INCLUSIONS = (
        "edge_slabs_around_vertices_inclusions_coefficient",
        r"$\mathcal{C}_{\mathrm{edge \ slabs, \ around \ vertices}}$",
        "slabs_around_vertices",
    )
    CONSTANT = ("constant_coefficient", r"$\mathcal{C}_{\mathrm{const}}$", "const")
    HETMANIUK_LEHOUCQ = (
        "hetmaniuk_lehoucq_coefficient",
        r"$\mathcal{C}_{\mathrm{hetmaniuk}}$",
        "hetmaniuk",
    )
    HEINLEIN = ("heinlein_coefficient", r"$\mathcal{C}_{\mathrm{heinlein}}$", "heinlein")

    def __init__(self, func_name: str, latex: str, short_name: str):
        self._func_name = func_name
        self._latex = latex
        self._short_name = short_name

    def get_func(self, obj: "DiffusionProblem") -> Callable:
        """Return the function name associated with the coefficient function."""
        return getattr(obj, self._func_name)

    @property
    def latex(self):
        """Return the LaTeX representation associated with the coefficient function."""
        return self._latex

    @property
    def short_name(self):
        """Return the short name associated with the coefficient function."""
        return self._short_name


class DiffusionProblem(Problem):
    """Class representing a diffusion problem on a two-level mesh with high contrast coefficients."""

    BACKGROUND_COEF = 1.0
    CONTRAST_COEF = 1e8

    def __init__(
        self,
        boundary_conditions: BoundaryConditions,
        mesh: TwoLevelMesh | MeshParams = DefaultQuadMeshParams.Nc4,
        coef_func: CoefFunc = CoefFunc.VERTEX_INCLUSIONS,
        source_func: SourceFunc = SourceFunc.CONSTANT,
        progress: Optional[PROGRESS] = None,
    ):
        self.progress = PROGRESS.get_active_progress_bar(progress)
        task = self.progress.add_task(f"Initializing DiffusionProblem ", total=3)
        LOGGER.info(
            f"Initializing DiffusionProblem for 1/H = {1 / mesh.coarse_mesh_size:.0f}"
        )

        # initialize the Problem with the TwoLevelMesh
        super().__init__(
            [boundary_conditions],
            mesh=mesh,
            ptype=ProblemType.DIFFUSION,
            progress=self.progress,
        )
        self.progress.advance(task)

        # construct finite element space
        self.construct_fespace()
        self.progress.advance(task)

        # get coefficient and source functions
        self.coef_func_name = coef_func.latex
        self.coef_func_short_name = coef_func.short_name
        self.coef_func = coef_func.get_func(self)()
        self.source_func_name = source_func.latex
        self.source_func_short_name = source_func.short_name
        self.source_func = source_func.get_func(self)()
        self.progress.advance(task)

        LOGGER.info("DiffusionProblem initialized successfully.")
        self.progress.soft_stop()

    def assemble(self, gfuncs=None):
        return super().assemble(gfuncs=[self.coef_func, self.source_func])

    ####################
    # source functions #
    ####################
    def constant_source(self):
        """Constant source function."""
        LOGGER.debug(f"Using constant source function.")
        return 1.0

    def parabolic_source(self):
        """Parabolic source function."""
        lx, ly = self.two_mesh.lx, self.two_mesh.ly
        LOGGER.debug(f"Using parabolic source function.")
        return 32 * (ngs.y * (ly - ngs.y) + ngs.x * (lx - ngs.x))

    #########################
    # coefficient functions #
    #########################
    def constant_coefficient(self):
        """Constant coefficient function."""
        LOGGER.debug(f"Using constant coefficient function.")
        return 1.0

    def hetmaniuk_lehoucq_coefficient(self):
        """Hetmaniuk & Lehoucq coefficient function. Equation 5.6 from Hetmaniuk & Lehoucq (2010)."""
        c = (
            1.2 + ngs.cos(32 * ngs.pi * ngs.x * (1 - ngs.x) * ngs.y * (1 - ngs.y))
        ) ** -1
        LOGGER.debug(f"Using Hetmaniuk & Lehoucq coefficient function (Eq. 5.6, 2010)")
        return c.Compile()

    def sinusoidal_coefficient(self):
        """Sinusoidal coefficient function. Equation 5.8 from Hetmaniuk & Lehoucq (2010)."""
        sx, sy = ngs.sin(25 * ngs.pi * ngs.x / self.two_mesh.lx), ngs.sin(
            25 * ngs.pi * ngs.y / self.two_mesh.ly
        )
        cy = ngs.cos(25 * ngs.pi * ngs.y / self.two_mesh.ly)
        c = ((2 + 1.8 * sx) / (2 + 1.8 * cy)) + ((2 + sy) / (2 + 1.8 * sx))
        LOGGER.debug(f"Using sinusoidal coefficient function (Eq. 5.8, 2010)")
        return c.Compile()

    def heinlein_coefficient(self):
        """Heinlein coefficient function. Equation 4.36 from Heinlein (2016)."""
        sx, sy = ngs.sin(25 * ngs.pi * ngs.x), ngs.sin(25 * ngs.pi * ngs.y)
        cy = ngs.cos(25 * ngs.pi * ngs.y)
        c = ((2 + 1.99 * sx) / (2 + 1.99 * cy)) + ((2 + sy) / (2 + 1.99 * sx))
        LOGGER.debug(f"Using Heinlein coefficient function (Eq. 4.36, 2016)")
        return c.Compile()

    def vertex_centered_inclusions_coefficient(self):
        """High coefficient inclusions around the coarse nodes."""
        # setup piecewise constant coefficient function
        constant_fes = ngs.L2(self.two_mesh.fine_mesh, order=0)
        grid_func = ngs.GridFunction(constant_fes)

        # Loop over coarse nodes and set high contrast on elements around them
        coef_array = np.full(self.two_mesh.fine_mesh.ne, self.BACKGROUND_COEF)
        for coarse_node in list(self.fes.free_component_tree_dofs.keys()):
            mesh_el = self.two_mesh.fine_mesh[coarse_node]
            for el in mesh_el.elements:
                coef_array[el.nr] = self.CONTRAST_COEF

        grid_func.vec.FV().NumPy()[:] = coef_array
        coef_func = ngs.CoefficientFunction(grid_func)
        LOGGER.debug(f"Using vertex centered inclusions coefficient function.")
        return coef_func.Compile()

    def two_layer_vertex_centered_inclusions_coefficient(self):
        """High coefficient inclusions around the coarse nodes with 2 layers of elements."""
        # setup piecewise constant coefficient function
        constant_fes = ngs.L2(self.two_mesh.fine_mesh, order=0)
        grid_func = ngs.GridFunction(constant_fes)

        # Get all free coarse node coordinates
        free_coarse_nodes = list(self.fes.free_component_tree_dofs.keys())

        # Loop over coarse nodes and set high contrast to 2 layers of elements around them
        coef_array = np.full(self.two_mesh.fine_mesh.ne, self.BACKGROUND_COEF)
        for coarse_node in free_coarse_nodes:
            mesh_el = self.two_mesh.fine_mesh[coarse_node]
            elements = set(mesh_el.elements)
            for el in elements:
                # add first layer
                coef_array[el.nr] = self.CONTRAST_COEF

                # get second layer elements
                outer_vertices = set(self.two_mesh.fine_mesh[el].vertices) - set(
                    [coarse_node]
                )
                for vertex in outer_vertices:
                    # filter inner elements
                    outer_elements = (
                        set(self.two_mesh.fine_mesh[vertex].elements) - elements
                    )
                    for outer_el in outer_elements:
                        # add second layer
                        coef_array[outer_el.nr] = self.CONTRAST_COEF

        grid_func.vec.FV().NumPy()[:] = coef_array
        coef_func = ngs.CoefficientFunction(grid_func)
        LOGGER.debug(
            f"Using two layer vertex centered inclusions coefficient function."
        )
        return coef_func.Compile()

    def three_layer_vertex_centered_inclusions_coefficient(self):
        """High coefficient inclusions around the coarse nodes with 3 layers of elements."""
        # setup piecewise constant coefficient function
        constant_fes = ngs.L2(self.two_mesh.fine_mesh, order=0)
        grid_func = ngs.GridFunction(constant_fes)

        # Get all free coarse node coordinates
        free_coarse_nodes = list(self.fes.free_component_tree_dofs.keys())

        # Loop over coarse nodes and set high contrast to 3 layers of elements around them
        coef_array = np.full(self.two_mesh.fine_mesh.ne, self.BACKGROUND_COEF)
        for coarse_node in free_coarse_nodes:
            mesh_el = self.two_mesh.fine_mesh[coarse_node]
            elements_first_layer = set(mesh_el.elements)
            for element_first_layer in elements_first_layer:
                coef_array[element_first_layer.nr] = self.CONTRAST_COEF

                # get second layer elements
                outer_vertices_first_layer = set(
                    self.two_mesh.fine_mesh[element_first_layer].vertices
                ) - set([coarse_node])
                for outer_vertex_first_layer in outer_vertices_first_layer:
                    # filter inner elements
                    elements_second_layer = (
                        set(self.two_mesh.fine_mesh[outer_vertex_first_layer].elements)
                        - elements_first_layer
                    )
                    for element_second_layer in elements_second_layer:
                        # add second layer
                        coef_array[element_second_layer.nr] = self.CONTRAST_COEF

                        # get third layer elements
                        outer_vertices_second_layer = (
                            set(self.two_mesh.fine_mesh[element_second_layer].vertices)
                            - outer_vertices_first_layer
                            - set([coarse_node])
                        )
                        for outer_vertex_second_layer in outer_vertices_second_layer:
                            # filter inner elements
                            elements_third_layer = (
                                set(
                                    self.two_mesh.fine_mesh[
                                        outer_vertex_second_layer
                                    ].elements
                                )
                                - elements_first_layer
                                - elements_second_layer
                            )
                            for element_third_layer in elements_third_layer:
                                # add third layer
                                coef_array[element_third_layer.nr] = self.CONTRAST_COEF

        grid_func.vec.FV().NumPy()[:] = coef_array
        coef_func = ngs.CoefficientFunction(grid_func)
        LOGGER.debug(
            f"Using two layer vertex centered inclusions coefficient function."
        )
        return coef_func.Compile()

    def edge_centered_inclusions_coefficient(self):
        """High coefficient inclusions centered on the coarse edges."""
        # setup piecewise constant coefficient function
        constant_fes = ngs.L2(self.two_mesh.fine_mesh, order=0)
        grid_func = ngs.GridFunction(constant_fes)

        # get num vertices on subdomain edges (without coarse nodes)
        num_edge_vertices = 2**self.two_mesh.refinement_levels - 2

        # set the indices of the edge vertices around which elements will be set to high contrast
        edge_vertex_inclusion_idxs = np.array(
            [i for i in range(2, num_edge_vertices, 2)]
        )

        # asssemble grid function
        coef_array = np.full(self.two_mesh.fine_mesh.ne, self.BACKGROUND_COEF)
        for coarse_edge in self.fes.free_coarse_edges:
            # get already sorted fine vertices on free coarse edge
            fine_vertices = self.two_mesh.coarse_edges_map[coarse_edge]["fine_vertices"]
            for i in edge_vertex_inclusion_idxs:
                vertex = fine_vertices[i]
                mesh_el = self.two_mesh.fine_mesh[vertex]
                for el in mesh_el.elements:
                    coef_array[el.nr] = self.CONTRAST_COEF

        grid_func.vec.FV().NumPy()[:] = coef_array
        coef_func = ngs.CoefficientFunction(grid_func)
        LOGGER.debug(f"Using edge centered inclusions coefficient function.")
        return coef_func.Compile()

    def single_slab_edge_inclusions_coefficient(self):
        """High coefficient inclusions centered on the coarse edges with slabs."""
        # setup piecewise constant coefficient function
        constant_fes = ngs.L2(self.two_mesh.fine_mesh, order=0)
        grid_func = ngs.GridFunction(constant_fes)

        # construct the coefficient array
        coef_array = np.full(self.two_mesh.fine_mesh.ne, self.BACKGROUND_COEF)
        for coarse_edge in self.fes.free_coarse_edges:
            slab_elements = self.two_mesh.edge_slabs[coarse_edge.nr]["single"]
            coef_array[slab_elements] = self.CONTRAST_COEF

        grid_func.vec.FV().NumPy()[:] = coef_array
        coef_func = ngs.CoefficientFunction(grid_func)
        LOGGER.debug(f"Using single slab edge inclusions coefficient function.")
        return coef_func.Compile()

    def double_slab_edge_inclusions_coefficient(self):
        # setup piecewise constant coefficient function
        constant_fes = ngs.L2(self.two_mesh.fine_mesh, order=0)
        grid_func = ngs.GridFunction(constant_fes)

        # construct the coefficient array
        coef_array = np.full(self.two_mesh.fine_mesh.ne, self.BACKGROUND_COEF)
        for coarse_edge in self.fes.free_coarse_edges:
            slab_elements = self.two_mesh.edge_slabs[coarse_edge.nr]["double"]
            coef_array[slab_elements] = self.CONTRAST_COEF

        grid_func.vec.FV().NumPy()[:] = coef_array
        coef_func = ngs.CoefficientFunction(grid_func)
        LOGGER.debug(f"Using double slab edge inclusions coefficient function.")
        return coef_func.Compile()

    def edge_slabs_around_vertices_inclusions_coefficient(self):
        constant_fes = ngs.L2(self.two_mesh.fine_mesh, order=0)
        grid_func = ngs.GridFunction(constant_fes)

        # construct the coefficient array
        coef_array = np.full(self.two_mesh.fine_mesh.ne, self.BACKGROUND_COEF)
        for coarse_node in self.fes.free_component_tree_dofs.keys():
            slab_elements = self.two_mesh.edge_slabs["around_coarse_nodes"][
                coarse_node.nr
            ]
            coef_array[slab_elements] = self.CONTRAST_COEF

        grid_func.vec.FV().NumPy()[:] = coef_array
        coef_func = ngs.CoefficientFunction(grid_func)
        LOGGER.debug(
            f"Using edge slabs around vertices inclusions coefficient function."
        )
        return coef_func.Compile()

    def save_functions(self):
        """Save the source and coefficient functions to vtk."""
        self.save_ngs_functions(
            funcs=[self.source_func, self.coef_func, self.u],
            names=[self.source_func_short_name, self.coef_func_short_name, "solution"],
            category="diffusion",
        )


class DiffusionProblemExample:
    # mesh parameters
    mesh_params = DefaultQuadMeshParams.Nc4  # DefaultQuadMeshParams.Nc4

    # source and coefficient functions
    source_func = SourceFunc.CONSTANT
    coef_func = CoefFunc.THREE_LAYER_VERTEX_INCLUSIONS

    # preconditioner and coarse space
    preconditioner = TwoLevelSchwarzPreconditioner  # TwoLevelSchwarzPreconditioner
    coarse_space = AMSCoarseSpace  # GDSWCoarseSpace or RGDSWCoarseSpace

    # use GPU for solving (not working yet)
    use_gpu = False

    # save coarse bases
    save_coarse_bases = False

    # save CG convergence information
    get_cg_info = True

    # save source, coefficient and solution as VTK files
    save_functions_toggle = True

    @classmethod
    def example_construction(cls):
        # create diffusion problem
        cls.diffusion_problem = DiffusionProblem(
            HomogeneousDirichlet(ProblemType.DIFFUSION),
            mesh=cls.mesh_params,
            source_func=cls.source_func,
            coef_func=cls.coef_func,
        )

    @classmethod
    def example_solve(cls):
        # solve problem
        cls.diffusion_problem.solve(
            preconditioner=cls.preconditioner,
            coarse_space=cls.coarse_space,
            rtol=1e-8,
            use_gpu=cls.use_gpu,
            save_cg_info=cls.get_cg_info,
            save_coarse_bases=cls.save_coarse_bases,
        )

    @classmethod
    def save_functions(cls):
        """Save the source and coefficient functions to vtk."""
        if cls.save_functions_toggle:
            cls.diffusion_problem.save_functions()

    @classmethod
    def visualize_convergence(cls):
        if cls.get_cg_info:
            import matplotlib.pyplot as plt
            import numpy as np

            from hcmsfem.plot_utils import set_mpl_cycler, set_mpl_style

            set_mpl_style()
            set_mpl_cycler(colors=True, lines=True)

            fig, axs = plt.subplots(
                2,
                2,
                figsize=(10, 6),
                gridspec_kw={"height_ratios": [3, 1], "width_ratios": [1, 1]},
            )

            # Remove the bottom-right axis and make the bottom-left axis span both columns
            fig.delaxes(axs[1, 0])
            fig.delaxes(axs[1, 1])
            gs = axs[1, 0].get_gridspec()
            axs_bottom = fig.add_subplot(gs[1, :])

            # plot the coefficients
            axs[0, 0].plot(cls.diffusion_problem.cg_alpha, label=r"$\alpha$")
            axs[0, 0].plot(cls.diffusion_problem.cg_beta, label=r"$\beta$")
            axs[0, 0].set_xlabel("Iteration")
            axs[0, 0].set_ylabel("Coefficient Value")
            axs[0, 0].legend()

            # plot residuals and preconditioned residuals
            axs[0, 1].plot(
                cls.diffusion_problem.cg_residuals, label=r"$||r_m||_2 / ||r_0||_2$"
            )
            if cls.diffusion_problem.cg_precond_residuals is not None:
                axs[0, 1].plot(
                    cls.diffusion_problem.cg_precond_residuals,
                    label=r"$||z_m||_2 / ||z_0||_2$",
                )
            axs[0, 1].set_xlabel("Iteration")
            axs[0, 1].set_ylabel("Relative residuals")
            axs[0, 1].set_yscale("log")
            axs[0, 1].legend()

            plt.suptitle(
                f"CG convergence ("
                + r"$\mathcal{C}$"
                + f" = {cls.coef_func.latex}, "
                + r"$f$"
                + f" = {cls.source_func.latex}, "
                + r"$M^{-1}$"
                + f" = {cls.diffusion_problem.precond_name})"
            )

            # Plot each spectrum at a different y
            y = 0
            axs_bottom.plot(
                np.real(cls.diffusion_problem.approximate_eigs),
                np.full_like(cls.diffusion_problem.approximate_eigs, y),
                marker="x",
                linestyle="None",
            )

            # Set y-ticks and labels
            axs_bottom.set_ylim(-0.5, 0.5)
            axs_bottom.set_yticks(
                [y],
                ["$\\mathbf{\\sigma(T_m)}$"],
            )
            axs_bottom.set_xscale("log")
            axs_bottom.grid(axis="x")
            axs_bottom.grid()
            ax2 = axs_bottom.twinx()

            # add condition numbers on right axis
            def format_cond(c):
                if np.isnan(c):
                    return "n/a"
                mantissa, exp = f"{c:.1e}".split("e")
                exp = int(exp)
                return rf"${mantissa} \times 10^{{{exp}}}$"

            cond = np.max(cls.diffusion_problem.approximate_eigs) / np.min(
                cls.diffusion_problem.approximate_eigs
            )
            ax2.set_ylim(axs_bottom.get_ylim())
            ax2.set_yticks([y], [format_cond(cond)])

            plt.tight_layout()
            plt.show()

    @classmethod
    def get_save_dir(cls):
        """Get the directory where the problem files are saved."""
        return TwoLevelMesh.get_save_dir(cls.mesh_params)


def full_example():
    DiffusionProblemExample.example_construction()
    DiffusionProblemExample.example_solve()
    DiffusionProblemExample.save_functions()
    DiffusionProblemExample.visualize_convergence()


def profile_construction():
    import cProfile
    import pstats

    from hcmsfem.visualize_profile import visualize_profile

    fp = DiffusionProblemExample.get_save_dir() / "diffusion_problem_construction.prof"
    cProfile.run("DiffusionProblemExample.example_construction()", str(fp))
    p = pstats.Stats(str(fp))
    p.sort_stats("cumulative").print_stats(10)
    visualize_profile(fp)


def profile_solve():
    import cProfile
    import pstats

    from hcmsfem.visualize_profile import visualize_profile

    DiffusionProblemExample.example_construction()
    fp = DiffusionProblemExample.get_save_dir() / "diffusion_problem_solve.prof"
    cProfile.run("DiffusionProblemExample.example_solve()", str(fp))
    p = pstats.Stats(str(fp))
    p.sort_stats("cumulative").print_stats(10)
    visualize_profile(fp)


if __name__ == "__main__":
    LOGGER.setLevel(LOGGER.INFO)
    # PROGRESS.turn_off()
    full_example()  # Uncomment this line to run a full diffusion problem example
    # profile_construction()  # Uncomment this line to profile problem construction
    # profile_solve()  # Uncomment this line to profile problem solving
