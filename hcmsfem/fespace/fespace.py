import json
from typing import Optional

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp

from hcmsfem.boundary_conditions import BoundaryConditions, HomogeneousDirichlet
from hcmsfem.logger import LOGGER, PROGRESS
from hcmsfem.meshes import DefaultQuadMeshParams, TwoLevelMesh
from hcmsfem.problem_type import ProblemType


class FESpace:
    """
    FESpace
    A class for constructing and managing a finite element space on a TwoLevelMesh,
    including the classification of DOFs (degrees of freedom) into interior, edge, and coarse node categories.
    """

    def __init__(
        self,
        two_mesh: TwoLevelMesh,
        boundary_conditions: list[BoundaryConditions],
        ptype: ProblemType = ProblemType.CUSTOM,
        progress: Optional[PROGRESS] = None,
    ):
        """
        Initialize the finite element space for the given mesh.

        Args:
            - two_mesh (TwoLevelMesh): The mesh to create the finite element space for.
            boundary_conditions (list[BoundaryConditions]): List of boundary conditions to apply. The order of the boundary conditions
            should match the unknowns in the problem type.
            ptype (ProblemType): The problem type defining the finite element space.
        Raises:
            - ValueError: If the sum of classified free DOFs does not match the total number of free DOFs in the space.
        """
        self.progress = PROGRESS.get_active_progress_bar(progress)
        task = self.progress.add_task(f"Constructing FESpace", total=2)
        LOGGER.info(f"Constructing FESpace for {str(ptype)}")

        self.two_mesh = two_mesh
        self.ptype = ptype
        self.ndofs_per_unknown = []

        # construct fespace to get dofs
        for fespace, order, dim, bcs in zip(
            ptype.fespaces, ptype.orders, ptype.dimensions, boundary_conditions
        ):
            fespace = fespace(two_mesh.fine_mesh, order=order, **bcs.boundary_kwargs)
            for _ in range(dim):
                self.ndofs_per_unknown.append(fespace.ndof // dim)
            if hasattr(self, "fespace"):
                self.fespace *= fespace
            else:
                self.fespace = fespace
        self.progress.advance(task)

        # calculating DOFs
        self.calculate_dofs()
        self.progress.advance(task)

        # check
        if self.num_free_dofs != (
            self.num_interior_dofs
            + self.num_coarse_node_dofs
            + self.num_interface_edge_dofs
            + self.num_interface_face_dofs
        ):
            msg = (
                f"Mismatch in number of free DOFs: {self.num_free_dofs} != "
                f"{self.num_interior_dofs} + {self.num_coarse_node_dofs} + "
                f"{self.num_interface_edge_dofs} + {self.num_interface_face_dofs}"
            )
            LOGGER.error(msg)
            raise ValueError(msg)

        LOGGER.info(f"FESpace constructed successfully")
        LOGGER.debug(str(self))
        self.progress.soft_stop()

    def calculate_dofs(self):
        """
        Calculate and classify the degrees of freedom (DOFs) in the finite element space.

        This method:
            - Collects all DOFs from the fine mesh.
            - Calls calculate_subdomain_dofs() to classify DOFs by subdomain.
            - Removes edge and coarse node DOFs from the set of interior DOFs.
            - Stores the classified DOFs and prints a summary.
        """
        # all DOFS
        self.total_dofs = self.fespace.ndof

        # free DOFs mask
        self.free_dofs_mask = np.array(self.fespace.FreeDofs()).astype(bool)
        self.num_free_dofs = np.sum(self.free_dofs_mask)

        # Map free dof global idx -> restricted idx
        self.free_idx = np.flatnonzero(self.free_dofs_mask)

        # calculate subdomain DOFs
        self.domain_dofs = self.get_subdomain_dofs()

        # component dofs
        self.free_coarse_node_dofs = self.get_free_coarse_node_dofs()
        self.free_edge_component_dofs, self.free_coarse_edges = (
            self.get_free_edge_component_dofs()
        )
        self.free_face_component_dofs = self.get_free_face_component_dofs()
        self.free_component_tree_dofs, self.edge_component_multiplicities = (
            self.get_free_component_tree_dofs()
        )
        # create a mask for free interface dofs
        interface_dofs = []
        for component_dofs in self.free_coarse_node_dofs:
            interface_dofs.extend(component_dofs)
        num_interface_edge_dofs = 0
        for component_dofs in self.free_edge_component_dofs:
            interface_dofs.extend(component_dofs)
            num_interface_edge_dofs += len(component_dofs)
        num_interface_face_dofs = 0
        for component_dofs in self.free_face_component_dofs:
            interface_dofs.extend(component_dofs)
            num_interface_face_dofs += len(component_dofs)
        self.interface_dofs_mask = np.zeros(self.fespace.ndof, dtype=bool)
        self.interface_dofs_mask[list(interface_dofs)] = True
        self.interface_dofs_mask = np.logical_and(
            self.free_dofs_mask, self.interface_dofs_mask
        )[self.free_dofs_mask]

        # create mask for interior dofs
        self.interior_dofs_mask = ~self.interface_dofs_mask

        # meta info
        self.num_coarse_node_dofs = len(self.free_coarse_node_dofs)
        self.num_interface_edge_dofs = num_interface_edge_dofs
        self.num_interface_face_dofs = num_interface_face_dofs
        self.num_interior_dofs = np.sum(self.interior_dofs_mask)

    def get_subdomain_dofs(self):
        """
        Calculate DOFs for each subdomain (coarse element) and classify them as interior, edge, coarse node, or layer DOFs.

        Returns:
            dict: Mapping from coarse elements to their classified DOFs.
        """
        domain_dofs = {}
        if self.domain_dofs_path.exists():
            return self.load_domain_dofs()
        else:
            task = self.progress.add_task(
                "Obtaining subdomain DOFs", total=len(self.two_mesh.subdomains)
            )
            desc = self.progress.get_description(task)
            desc += " ({0})"
            for subdomain, subdomain_data in self.two_mesh.subdomains.items():
                # all subdomain dofs
                dofs = []

                # interior dofs
                for el in subdomain_data["interior"]:
                    dofs.extend(self.fespace.GetDofNrs(el))

                # get all (unique) layer dofs that are not on the boundary
                for layer_idx in range(1, self.two_mesh.layers + 1):
                    for el in subdomain_data[f"layer_{layer_idx}"]:
                        dofs.extend(self.fespace.GetDofNrs(el))

                # get all dofs
                dofs = np.unique(dofs).tolist()

                domain_dofs[subdomain] = dofs

                # update PROGRESS bar
                self.progress.update(
                    task,
                    advance=1,
                    description=desc.format(subdomain.nr),
                )

            # remove task and log completion
            self.progress.remove_task(task)
            LOGGER.debug("Obtained subdomain DOFs")  # type: ignore

            # save domain dofs to file for later use
            self.save_domain_dofs(domain_dofs)
        return domain_dofs

    def get_free_coarse_node_dofs(self) -> list[list[int]]:
        """
        Get the degrees of freedom (DOFs) associated with coarse nodes.

        Returns:
            list: A list of DOFs corresponding to coarse nodes.
        """
        coarse_node_dofs = []
        all_dofs = np.arange(self.fespace.ndof)
        free_dofs = all_dofs[self.free_dofs_mask]
        total = len(self.two_mesh.connected_components["coarse_nodes"])
        task = self.progress.add_task(
            "Obtaining free coarse node DOFs",
            total=total,
        )
        desc = self.progress.get_description(task)
        desc += " ({0})"
        for coarse_node in self.two_mesh.connected_components["coarse_nodes"]:
            dofs = list(self.fespace.GetDofNrs(coarse_node))
            coarse_node_dofs.append(dofs[0])
            self.progress.update(
                task,
                advance=1,
                description=desc.format(coarse_node),
            )

        # filter coarse node dofs to only include free dofs
        coarse_node_dofs_arr = np.array(coarse_node_dofs)
        coarse_node_dofs = coarse_node_dofs_arr[
            np.isin(coarse_node_dofs_arr, free_dofs)
        ]

        # convert to list of lists (desired output format, even if c is only one DOF)
        coarse_node_dofs = [[c] for c in coarse_node_dofs]

        # remove task and log completion
        self.progress.remove_task(task)
        LOGGER.debug("Obtained free coarse node DOFs")
        return coarse_node_dofs

    def get_free_edge_component_dofs(self) -> tuple[list[list[int]], list[ngs.NodeId]]:
        edge_component_dofs = []
        all_dofs = np.arange(self.fespace.ndof)
        free_dofs = all_dofs[self.fespace.FreeDofs()]
        coarse_edges = []
        dofs_arrs = []
        task = self.progress.add_task(
            "Obtaining free edge component DOFs",
            total=len(self.two_mesh.connected_components["edges"]),
        )
        desc = self.progress.get_description(task)
        desc += " ({0})"
        for coarse_edge, edge_component in self.two_mesh.connected_components[
            "edges"
        ].items():
            coarse_edges.append(coarse_edge)
            dofs = []
            for c in edge_component:
                dofs.extend(self.fespace.GetDofNrs(c))

            # remove boundary dofs
            dofs_arrs.append(np.array(dofs))

            self.progress.update(
                task,
                advance=1,
                description=desc.format(coarse_edge),
            )

        # collect all dofs
        dofs_arrs = np.array(dofs_arrs)
        num_edge_dofs_per_edge: int = dofs_arrs.shape[1]

        # filter dofs to only include free dofs
        isin = np.isin(dofs_arrs, free_dofs)
        any_free_dofs = np.any(isin, axis=1)
        num_free_dofs = np.sum(any_free_dofs)
        dofs_arrs = dofs_arrs[isin].reshape(num_free_dofs, num_edge_dofs_per_edge)
        for dofs in dofs_arrs:
            edge_component_dofs.append(dofs.tolist())

        # get list of free coarse edges
        coarse_edges = np.array(coarse_edges)
        free_coarse_edges = coarse_edges[any_free_dofs].tolist()

        # remove task and log completion
        self.progress.remove_task(task)
        LOGGER.debug("Obtained free edge component DOFs")
        return edge_component_dofs, free_coarse_edges

    def get_free_face_component_dofs(self) -> list[list[int]]:
        face_component_dofs = []
        all_dofs = np.arange(self.fespace.ndof)
        free_dofs = all_dofs[self.fespace.FreeDofs()]
        task = self.progress.add_task(
            "Obtaining free face component DOFs",
            total=len(self.two_mesh.connected_components["faces"]),
        )
        for face_component in self.two_mesh.connected_components["faces"].values():
            dofs = []
            for c in face_component:
                dofs.extend(self.fespace.GetDofNrs(c))

            # remove boundary dofs
            dofs_arr = np.array(dofs)
            dofs = dofs_arr[np.isin(dofs_arr, free_dofs)].tolist()

            if len(dofs) > 0:
                face_component_dofs.append(dofs)

            self.progress.advance(task)

        # remove task and log completion
        self.progress.remove_task(task)
        LOGGER.debug("Obtained free face component DOFs")  # type: ignore
        return face_component_dofs

    def get_free_component_tree_dofs(self):
        """
        Traverse the connected component tree and collect all DOFs associated with free
        coarse nodes also keep track of how many times a connected edge component appears
        (its multiplicity).

        The result is a dictionary very similar to the component tree, only now it looks like this:
        {
            coarse_node_i1: {
                "node": int, # DOF of the coarse node
                "edges" : {
                    coarse_edge_j1: [dof1, dof2, ...], # DOFs of the coarse edge
                    coarse_edge_j2: {...}
                    ...
                    coarse_edge_jk: {...}
                    }
            },
            coarse_node_i2: {
                coarse_edge_j1: {...},
                ...
            },
        })
        """
        all_dofs = np.arange(self.fespace.ndof)
        free_dofs = all_dofs[self.fespace.FreeDofs()]
        free_component_tree_dofs = {}
        edge_component_multiplicity = {}
        component_tree = self.two_mesh.connected_component_tree
        task = self.progress.add_task(
            "Obtaining free component tree DOFs",
            total=len(component_tree),
        )
        for coarse_node, components in component_tree.items():
            coarse_node_dofs = list(self.fespace.GetDofNrs(coarse_node))
            if coarse_node_dofs in self.free_coarse_node_dofs:
                free_component_tree_dofs[coarse_node] = {"node": coarse_node_dofs}
            else:
                continue  # skip coarse nodes without free DOFs

            # treat coarse edges
            free_component_tree_dofs[coarse_node]["edges"] = {}

            for coarse_edge, edge_components in components["edges"].items():
                # increment the multiplicity of the coarse edge component
                edge_component_multiplicity[coarse_edge] = (
                    edge_component_multiplicity.get(coarse_edge, 0) + 1
                )

                # get all DOFs of the edge component (free by construction)
                dofs = []
                for c in edge_components:
                    edge_dofs = list(self.fespace.GetDofNrs(c))
                    dofs.extend(edge_dofs)
                free_component_tree_dofs[coarse_node]["edges"][coarse_edge] = dofs

            self.progress.advance(task)

        LOGGER.debug("Obtained free component tree DOFs and edge multiplicities")
        self.progress.remove_task(task)
        return free_component_tree_dofs, edge_component_multiplicity

    def map_global_to_restricted_dofs(self, global_dofs: np.ndarray) -> np.ndarray:
        """Efficiently get free dofs as the intersection of global and free dofs"""
        free_global = global_dofs[self.free_dofs_mask[global_dofs]]

        # Map global free dofs to restricted matrix indices
        return np.searchsorted(self.free_idx, free_global)

    def get_prolongation_operator(self):
        """
        Get the prolongation operator for the finite element space.

        Returns:
            ngs.ProlongationOperator: The prolongation operator for the finite element space.
        """
        prolongation_operator = self.fespace.Prolongation().Operator(1)
        for level_idx in range(1, self.two_mesh.refinement_levels):
            prolongation_operator = (
                self.fespace.Prolongation().Operator(level_idx + 1)
                @ prolongation_operator
            )
        return sp.csc_matrix(prolongation_operator.ToDense().NumPy())

    def get_gridfunc(self, vals):
        """
        Get the grid function representing the DOFs on the mesh.

        Args:
            vals (np.ndarray): The values to plot at each DOF.

        Returns:
            ngs.GridFunction: The grid function representing the DOFs on the mesh.
        """
        vals = np.asarray(vals, dtype=float).flatten()
        assert len(vals) == self.fespace.ndof, LOGGER.error(
            f"Length of vals ({len(vals)}) does not match number of DOFs ({self.fespace.ndof})"
        )
        grid_function = ngs.GridFunction(self.fespace)
        grid_function.vec.FV().NumPy()[:] = vals
        return grid_function

    def save_domain_dofs(self, domain_dofs: dict[ngs.ELEMENT, list[int]]):
        """
        Save the domain DOFs to a JSON file.

        This method saves the classified DOFs for each subdomain to a file in the save directory of the TwoLevelMesh.
        """
        domain_dofs_data = {
            subdomain.nr: dofs for subdomain, dofs in domain_dofs.items()
        }
        with open(self.domain_dofs_path, "w") as f:
            json.dump(domain_dofs_data, f, indent=4)
        LOGGER.debug("Saved domain DOFs to  %s", self.domain_dofs_path)

    def load_domain_dofs(self):
        """
        Load the domain DOFs from a JSON file.

        This method loads the classified DOFs for each subdomain from a file in the save directory of the TwoLevelMesh.
        """
        domain_dofs = {}
        with open(self.domain_dofs_path, "r") as f:
            domain_dofs_data = json.load(f)
        for subdomain_nr, dofs in domain_dofs_data.items():
            el_id = ngs.ElementId(int(subdomain_nr))
            domain_dofs[self.two_mesh.coarse_mesh[el_id]] = dofs
        LOGGER.debug(f"Loaded domain DOFs from %s", self.domain_dofs_path)
        return domain_dofs

    @property
    def domain_dofs_path(self):
        """
        Get the path to the domain DOFs file.

        Returns:
            str: The path to the domain DOFs file.
        """
        return self.two_mesh.save_dir / f"domain_dofs_{self.ptype.name}.json"

    @property
    def u(self):
        """
        Get the trial function of the finite element space.

        Returns:
            ngs.TrialFunction: The trial function of the finite element space.
        """
        return self.fespace.TrialFunction()

    @property
    def v(self):
        """
        Get the test function of the finite element space.

        Returns:
            ngs.TestFunction: The test function of the finite element space.
        """
        return self.fespace.TestFunction()

    def __repr__(self):
        repr_str = "FE space DOFs:"
        repr_str += f"\n\ttotal DOFs: {self.total_dofs}"
        repr_str += f"\n\tfespace dimension: {'X'.join([str(d) for d in self.ptype.dimensions])}"
        repr_str += f"\n\tinterior DOFs: {self.num_interior_dofs}"
        repr_str += f"\n\tedge DOFs: {self.num_interface_edge_dofs}"
        repr_str += f"\n\tface DOFs: {self.num_interface_face_dofs}"
        repr_str += f"\n\tcoarse Node DOFs: {self.num_coarse_node_dofs}"
        return repr_str


def load_mesh():
    return TwoLevelMesh.load(DefaultQuadMeshParams.Nc4)


def example_fespace():
    """
    Example function to create a finite element space on a TwoLevelMesh.
    """
    two_mesh = load_mesh()
    ptype = ProblemType.DIFFUSION
    fespace = FESpace(two_mesh, [HomogeneousDirichlet(ptype)], ptype)


def profile_fespace():
    """
    Profile the FESpace class to analyze its performance.
    """
    import cProfile
    import pstats

    from hcmsfem.visualize_profile import visualize_profile

    two_mesh = load_mesh()
    ptype = ProblemType.DIFFUSION
    fp = two_mesh.save_dir / "fespace_profile.prof"
    cProfile.runctx(
        "fespace = FESpace(two_mesh, [HomogeneousDirichlet(ptype)], ptype)",
        globals(),
        {"ptype": ptype, "two_mesh": two_mesh},
        str(fp),
    )
    p = pstats.Stats(str(fp))
    p.sort_stats("cumulative").print_stats(10)
    visualize_profile(fp)


if __name__ == "__main__":
    example_fespace()  # Uncomment to run the example
    # profile_fespace()  # Uncomment to profile the FESpace class
