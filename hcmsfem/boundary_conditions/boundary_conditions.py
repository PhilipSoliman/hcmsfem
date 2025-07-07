from enum import Enum

import ngsolve as ngs

from hcmsfem.logger import LOGGER
from hcmsfem.meshes import BoundaryName
from hcmsfem.problem_type import ProblemType


class BoundaryType(Enum):
    """
    BoundaryType
    A class representing the type of boundary condition for a PDE problem.
    """

    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"


class BoundaryCondition:
    """Class representing a boundary condition for a problem."""

    def __init__(
        self,
        name: BoundaryName,
        btype: BoundaryType,
        values: dict[int, float | ngs.CoefficientFunction],
    ):
        """Constructor for the BoundaryCondition class.

        Args:
            name (BoundaryName): The name of the boundary.
            boundary_type (BoundaryType): The type of the boundary condition.
            values (dict[int, float | ngs.CoefficientFunction]): The values of the boundary condition.
                The keys are the component indices, and the values are the corresponding values.
                For example, {0: 1.0, 1: 2.0} means that component 0 has a value of 1.0 and component 1 has a value of 2.0.
        """
        self.name = name
        self.type = btype
        self.values = values


class BoundaryConditions:
    """Class representing a boundary condition for a problem."""

    MISSING_BC_ERROR = "Boundary conditions not set for all boundaries. Missing {0}."

    def __init__(self):
        """Constructor for the BoundaryConditions class.

        Args:
            name (BoundaryName): The name of the boundary.
            boundary_type (BoundaryType): The type of the boundary condition.
        """
        self._boundary_conditions = []
        self._boundary_kwargs = {}
        self._unset_boundaries = [bname for bname in BoundaryName]

    def set_boundary_condition(self, bc: BoundaryCondition):
        """
        Set the boundary conditions for the problem.

        Args:
            conditions (list[BoundaryCondition]): List of boundary conditions.
        """
        # check for previously defined bc
        old_bc = next((b for b in self._boundary_conditions if b.name == bc.name), None)
        if old_bc:
            LOGGER.debug(
                f"Replacing existing boundary condition on {bc.name.value} boundary"
            )
            self._boundary_conditions.remove(old_bc)
        else:
            self._unset_boundaries.remove(bc.name)

        self._boundary_conditions.append(bc)

    @property
    def boundary_kwargs(self):
        # if self._unset_boundaries != []:
        #     raise ValueError(self.MISSING_BC_ERROR.format(self._unset_boundaries))

        boundary_kwargs = {}
        for bc in self._boundary_conditions:
            if bc.type.value in boundary_kwargs:
                boundary_kwargs[bc.type.value] += f"|{bc.name.value}"
            else:
                boundary_kwargs[bc.type.value] = f"{bc.name.value}"

        return boundary_kwargs

    def set_boundary_conditions_on_gfunc(
        self, u: ngs.GridFunction, fespace: ngs.FESpace, mesh: ngs.Mesh
    ):
        for bc in self._boundary_conditions:
            self.set_boundary_condition_on_gfunc(u, bc, fespace, mesh)

    def set_boundary_condition_on_gfunc(
        self,
        u: ngs.GridFunction,
        bc: BoundaryCondition,
        fespace: ngs.FESpace,
        mesh: ngs.Mesh,
    ):
        """Set a single boundary condition on a grid function."""
        if bc.type == BoundaryType.DIRICHLET:
            u_tmp = ngs.GridFunction(fespace)
            if len(bc.values) > 1:
                for component, value in bc.values.items():
                    u_tmp.components[component].Set(
                        value,
                        definedon=mesh.Boundaries(bc.name.value),
                    )
                    LOGGER.info(
                        f"Set Dirichlet condition on {bc.name.value} boundary for component {component} to {value}"
                    )

            else:
                u_tmp.Set(bc.values[0], definedon=mesh.Boundaries(bc.name.value))
            u.vec.data += u_tmp.vec
        else:
            msg = (
                f"Boundary condition {bc.name.value} of type {bc.type.value} is not supported. "
                "Only Dirichlet conditions are supported."
            )
            LOGGER.error(msg)
            raise NotImplementedError(msg)

    def __repr__(self):
        out = "Boundary Conditions:"
        widths = [len(name.value) for name in self.names]
        width = max(widths)
        for name, btype, value in zip(self.names, self.btypes, self.values):
            out += f"\n\t{name.value:<{width}} {btype.value:<{width}} = {str(value):<{width}}"
        out += f"\n\tkwargs (fespace) = {self.boundary_kwargs}"
        return out

    @property
    def names(self):
        """Get the names of the boundaries."""
        return [bc.name for bc in self._boundary_conditions]

    @property
    def btypes(self):
        """Get the types of the boundaries."""
        return [bc.type for bc in self._boundary_conditions]

    @property
    def values(self) -> list[str]:
        """Get the values of the boundaries."""
        out = []
        for bc in self._boundary_conditions:
            comp_str = ", ".join(
                f"comp {component}: {value}" for component, value in bc.values.items()
            )
            out.append(comp_str)
        return out

    @property
    def num_bcs(self):
        """Get the number of boundaries."""
        return len(BoundaryName)


class HomogeneousDirichlet(BoundaryConditions):

    def __init__(self, ptype: ProblemType):
        super().__init__()
        values = {}
        if ptype == ProblemType.DIFFUSION:
            values = {0: 0.0}
        elif ptype == ProblemType.NAVIER_STOKES:
            values = {0: 0.0, 1: 0.0}  # u_x = 0.0, u_y = 0.0
        for bname in BoundaryName:
            self.set_boundary_condition(
                BoundaryCondition(
                    bname,
                    BoundaryType.DIRICHLET,
                    values,
                )
            )
        LOGGER.info(f"Set homogeneous Dirichlet")
        LOGGER.debug(str(self))
