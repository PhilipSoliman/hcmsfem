import copy
import json
from enum import Enum
from itertools import cycle
from pathlib import Path
from typing import Iterator, Literal, Optional

import matplotlib.pyplot as plt
import netgen.libngpy._NgOCC as occ
import ngsolve as ngs
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from ngsolve.meshes import MakeQuadMesh

from hcmsfem.logger import LOGGER, PROGRESS
from hcmsfem.plot_utils import CUSTOM_COLORS_SIMPLE
from hcmsfem.root import get_venv_root

DATA_DIR = get_venv_root() / "data"


def OCCRectangle(l, w):
    return occ.WorkPlane().Rectangle(l, w)


class BoundaryName(Enum):
    """
    Enum for boundary names used in the TwoLevelMesh class.
    """

    BOTTOM = "bottom"
    RIGHT = "right"
    TOP = "top"
    LEFT = "left"


class MeshParams:
    """
    Class for mesh parameters used in the TwoLevelMesh class.
    """

    def __init__(
        self,
        lx: float,
        ly: float,
        Nc: int,
        refinement_levels: int,
        layers: int,
        quad: bool = True,
    ):
        """
        Initialize the mesh parameters.
        Args:
            lx (float): Length of the domain in the x-direction.
            ly (float): Length of the domain in the y-direction.
            Nc (int): Number of coarse mesh cells along one dimension (H = 1/Nc).
            refinement_levels (int): Number of uniform refinements for the fine mesh.
            layers (int): Number of layers for domain decomposition.
        """
        self.lx = lx
        self.ly = ly
        self.coarse_mesh_size = 1 / Nc
        self.refinement_levels = refinement_levels
        self.layers = layers
        self.quad = quad


class DefaultMeshParamsMeta(type):
    def __iter__(cls) -> Iterator[MeshParams]:
        items = []
        for attr in dir(cls):
            if attr.startswith("Nc") and attr[2:].isdigit():
                value = getattr(cls, attr)
                if isinstance(value, MeshParams):
                    items.append((int(attr[2:]), value))
        for _, value in sorted(items):
            yield value

    def __len__(cls) -> int:
        return sum(
            1
            for attr in dir(cls)
            if attr.startswith("Nc")
            and attr[2:].isdigit()
            and isinstance(getattr(cls, attr), MeshParams)
        )


class DefaultQuadMeshParams(metaclass=DefaultMeshParamsMeta):
    Nc4 = MeshParams(lx=1.0, ly=1.0, Nc=4, refinement_levels=4, layers=2, quad=True)
    Nc8 = MeshParams(lx=1.0, ly=1.0, Nc=8, refinement_levels=4, layers=2, quad=True)
    Nc16 = MeshParams(lx=1.0, ly=1.0, Nc=16, refinement_levels=4, layers=2, quad=True)
    Nc32 = MeshParams(lx=1.0, ly=1.0, Nc=32, refinement_levels=4, layers=2, quad=True)
    Nc64 = MeshParams(lx=1.0, ly=1.0, Nc=64, refinement_levels=4, layers=2, quad=True)


class DefaultTriMeshParams(metaclass=DefaultMeshParamsMeta):
    Nc4 = MeshParams(lx=1.0, ly=1.0, Nc=4, refinement_levels=4, layers=2, quad=False)
    Nc8 = MeshParams(lx=1.0, ly=1.0, Nc=8, refinement_levels=4, layers=2, quad=False)
    Nc16 = MeshParams(lx=1.0, ly=1.0, Nc=16, refinement_levels=4, layers=2, quad=False)
    Nc32 = MeshParams(lx=1.0, ly=1.0, Nc=32, refinement_levels=4, layers=2, quad=False)
    Nc64 = MeshParams(lx=1.0, ly=1.0, Nc=64, refinement_levels=4, layers=2, quad=False)


class TwoLevelMesh:
    """
    TwoLevelMesh
    A class for generating, managing, and visualizing two-level conforming meshes (coarse and fine) for rectangular domains, with support for layering domain decomposition and mesh serialization.
    Attributes:
        lx (float): Length of the domain in the x-direction.
        ly (float): Length of the domain in the y-direction.
        coarse_mesh_size (float): Mesh size for the coarse mesh.
        refinement_levels (int): Number of uniform refinements for the fine mesh.
        fine_mesh (ngs.Mesh): Refined fine mesh.
        coarse_mesh (ngs.Mesh): Coarse mesh.
        subdomains (dict): Mapping of coarse elements to their corresponding fine mesh domains and layers.
        layers(int): Number of layers for domain decomposition.
    Methods:
        - __init__(lx, ly, coarse_mesh_size, refinement_levels=1, layers=0):
            Initialize the TwoLevelMesh with domain size, mesh size, refinement, and layers.
        - create_conforming_meshes():
            Create conforming coarse and fine meshes for the rectangular domain.
        - get_subdomains():
            Identify and return the mapping of coarse mesh elements to their corresponding fine mesh domains.
        - get_subdomain_edges(subdomain, interior_elements):
            Find all fine mesh edges that lie on the edges of a given coarse mesh element.
        - extend_subdomains(layer_idx=1):
            Extend the subdomains by adding layers of fine mesh elements.
        - save(file_name=""):
            Save the mesh, metadata, and subdomain information to disk.
        - load(file_name=""):
            Class method to load a TwoLevelMesh instance from saved files.
        - plot_domains(domains=1, plot_layers=False, opacity=0.9, fade_factor=1.5):
            Visualize the subdomains and optional layers using matplotlib.
        - plot_element(ax, element, mesh, fillcolor="blue", edgecolor="black", alpha=1.0, label=None):
            Static method to plot a single mesh element on a matplotlib axis.
    Usage:
        - Construct a TwoLevelMesh for a rectangular domain.
        - Save and load mesh and domain decomposition data.
        - Visualize mesh domains and layerss for debugging or analysis.
    """

    SAVE_STRING = "{0}_tlm_lx={1:.1f}_ly={2:.1f}_Nc={3:.0f}_lvl={4:.0f}_lyr={5:.0f}"
    ZORDERS = {
        "elements": 1.0,
        "layers": 1.5,
        "edges": 2.0,
        "vertices": 3.0,
    }

    def __init__(
        self,
        mesh_params: MeshParams = DefaultQuadMeshParams.Nc4,
        progress: Optional[PROGRESS] = None,
    ):
        """
        Initialize the TwoLevelMesh with domain size, mesh size, refinement, and layers.

        Args:
            lx (float): Length of the domain in the x-direction.
            ly (float): Length of the domain in the y-direction.
            coarse_mesh_size (float): Mesh size for the coarse mesh.
            refinement_levels (int, optional): Number of uniform refinements for the fine mesh. Defaults to 1.
            layers(int, optional): Number of layers for domain decomposition. Defaults to 0.
        """
        # handle input
        self.mesh_params = mesh_params
        self.lx = mesh_params.lx
        self.ly = mesh_params.ly
        self.coarse_mesh_size = mesh_params.coarse_mesh_size
        self.refinement_levels = mesh_params.refinement_levels
        self.layers = mesh_params.layers
        self.quad = mesh_params.quad

        # intial log
        self.progress = PROGRESS.get_active_progress_bar(progress)
        task = self.progress.add_task(f"Creating TwoLevelMesh", total=8)
        LOGGER.info(f"Creating TwoLevelMesh H = 1/{1/self.coarse_mesh_size:.0f}")

        # create mesh attributes
        self.fine_mesh = self.create_mesh()
        self.progress.advance(task)

        # get coarse mesh and edges map
        self.coarse_edges_map, self.coarse_mesh = (
            self._refine_mesh_and_get_coarse_edges_map(self.fine_mesh)
        )
        self.progress.advance(task)

        # calculate subdomains
        self.subdomains = self.get_subdomains()
        self.progress.advance(task)

        # get connected components
        self.connected_components = self.get_connected_components()
        self.progress.advance(task)

        self.connected_component_tree = self.get_connected_component_tree(
            self.connected_components
        )
        self.progress.advance(task)

        # extend subdomains and calculate overlapping subdomains
        self.extend_subdomains()
        self.progress.advance(task)

        # calculate edge slabs
        self.edge_slabs = self.generate_edge_slabs()
        self.progress.advance(task)

        LOGGER.info(f"Created TwoLevelMesh")
        LOGGER.debug(str(self))

        self.progress.soft_stop()

    # mesh creation
    def create_mesh(self) -> ngs.Mesh:
        """
        Create conforming coarse and fine meshes by refining the coarse mesh.
        Args:
            lx, ly: Rectangle dimensions.
            coarse_mesh_size: Mesh size for the coarse mesh.
            refinement_levels: Number of uniform refinements for the fine mesh.
        Returns:
            fine_mesh (ngs.Mesh): Refined NGSolve mesh.
            coarse_mesh (ngs.Mesh): Original coarse NGSolve mesh.
        """
        if self.quad:
            Nc = int(1 / self.coarse_mesh_size)
            mesh = MakeQuadMesh(nx=Nc, ny=Nc)  # bbpts=..., bbnames=...
        else:
            # Create rectangle geometry
            domain = OCCRectangle(self.lx, self.ly)
            face = domain.Face()

            # Set boundary labels (as before)
            face.edges.Max(occ.Y).name = BoundaryName.TOP.value
            face.edges.Min(occ.Y).name = BoundaryName.BOTTOM.value
            face.edges.Max(occ.X).name = BoundaryName.RIGHT.value
            face.edges.Min(occ.X).name = BoundaryName.LEFT.value

            geo = occ.OCCGeometry(face, dim=2)

            # Generate coarse mesh
            ngm_coarse = geo.GenerateMesh(
                minh=self.coarse_mesh_size,
                maxh=self.coarse_mesh_size,
            )
            mesh = ngs.Mesh(ngm_coarse)

        LOGGER.debug("created coarse mesh")
        return mesh

    ###################
    # mesh refinement #
    ###################
    def _refine_mesh_and_get_coarse_edges_map(
        self, mesh: ngs.Mesh
    ) -> tuple[dict, ngs.Mesh]:
        # Make a copy of the coarse mesh for easy reference (DO NOT REFINE THIS MESH... memory issues)
        coarse_mesh = copy.deepcopy(mesh)

        # initialize coarse edges dictionary
        num_fine_vertices_per_coarse_edge = 2**self.refinement_levels - 1
        coarse_edges_map = {
            coarse_edge: {
                "fine_vertices": np.empty(
                    num_fine_vertices_per_coarse_edge, dtype=ngs.NodeId
                ),
                "fine_edges": [],
            }
            for coarse_edge in coarse_mesh.edges
        }

        # precompute vertex masks for coarse edges
        vertex_masks = []
        start = (num_fine_vertices_per_coarse_edge - 1) // 2
        for ref_level in range(self.refinement_levels):
            step = 2 * (start + 1)
            vertex_mask = np.arange(start, num_fine_vertices_per_coarse_edge, step)
            vertex_masks.append(vertex_mask)
            start = (start - 1) // 2

        # get all coarse edge line segments
        num_coarse_edges = len(coarse_mesh.edges)

        refinement_task = self.progress.add_task(
            "refining mesh and processing coarse edges",
            total=self.refinement_levels,
        )
        for ref_level in range(self.refinement_levels):
            mesh.Refine()
            vertex_mask = vertex_masks[ref_level]
            edge_task = self.progress.add_task(
                f"processing coarse edges (level {ref_level + 1})",
                total=num_coarse_edges,
            )
            for coarse_edge, data in coarse_edges_map.items():
                # update coarse edges map with fine vertices
                coarse_edges_map[coarse_edge]["fine_vertices"][vertex_mask] = (
                    self._get_vertex_ids_on_coarse_edge(
                        coarse_edge, mesh, ref_level, data["fine_vertices"]
                    )
                )
                self.progress.advance(edge_task)
            self.progress.remove_task(edge_task)
            self.progress.advance(refinement_task)
            LOGGER.debug(
                f"Refined mesh at refinement level {ref_level + 1}/{self.refinement_levels}"
            )

        # remove the refinement task and log the completion
        self.progress.remove_task(refinement_task)
        LOGGER.debug("refined mesh and found fine vertices on coarse edges")

        # find fine edges on coarse edges using the vertices on coarse edges
        task = self.progress.add_task(
            "Finding fine edges on coarse edges",
            total=len(coarse_edges_map),
        )
        for coarse_edge, data in coarse_edges_map.items():
            # convert numpy array of fine vertices to list
            data["fine_vertices"] = list(data["fine_vertices"])

            # get all vertices on this coarse edge
            coarse_vertices = coarse_edge.vertices
            fine_and_coarse_vertices = (
                [coarse_vertices[0]] + data["fine_vertices"] + [coarse_vertices[1]]
            )
            data["fine_edges"] = self._get_edges_between_vertices(
                fine_and_coarse_vertices, mesh
            )

            self.progress.advance(task)
        self.progress.remove_task(task)
        LOGGER.debug("found fine edges on coarse edges")
        return coarse_edges_map, coarse_mesh

    def _get_vertex_ids_on_coarse_edge(
        self,
        coarse_edge: ngs.NodeId,
        fine_mesh: ngs.Mesh,
        ref_level: int,
        fine_vertices_on_edge,
    ):
        # get vertices on coarse edge
        vertices = coarse_edge.vertices
        if ref_level > 0:  # for refinement level 0, we only have coarse vertices
            fine_vertices = fine_vertices_on_edge[fine_vertices_on_edge != None]
            vertices = np.concatenate(([vertices[0]], fine_vertices, [vertices[1]]))  # type: ignore

        # get associated edges
        surrounding_edges = set()
        for v in vertices:
            surrounding_edges.update(set(fine_mesh[v].edges))

        # get associated vertex points and ids
        ids = []
        for e in surrounding_edges:
            fine_mesh_edge = fine_mesh[e]
            for v in fine_mesh_edge.vertices:
                if v not in coarse_edge.vertices:
                    ids.append(v.nr)

        # Find unique ids and their counts
        unique_ids, counts = np.unique(ids, return_counts=True)

        # vertices that appear exactly twice are on the coarse edge (+ possibly other ones)
        ids = unique_ids[counts == 2]

        # get points
        points = np.array(
            [fine_mesh[ngs.NodeId(ngs.VERTEX, vertex_id)].point for vertex_id in ids]
        )

        # determine whether the points are on the coarse edge line segment
        coarse_v0, coarse_v1 = coarse_edge.vertices
        coarse_p0, coarse_p1 = fine_mesh[coarse_v0].point, fine_mesh[coarse_v1].point
        coarse_edge_line = np.array([coarse_p0, coarse_p1])
        points_mask, distances = self._vectorized_check_if_point_on_line(
            points, coarse_edge_line
        )

        # filter ids based on the points that are on the coarse edge line segment
        ids = ids[points_mask.flatten()]

        # sort ids by distance to coarse vertex 0
        distances = distances[points_mask.flatten()]
        sorted_indices = np.argsort(distances.flatten())
        ids = ids[sorted_indices]

        # convert to ngs.NodeId
        ids = np.array([ngs.NodeId(ngs.VERTEX, vertex_id) for vertex_id in ids])

        return ids

    @staticmethod
    def _vectorized_check_if_point_on_line(points, lines, atol=1e-9):
        """
        Vectorized check if points are on line segments.

        Args:
            points (np.ndarray): (N, D) array of points.
            lines (np.ndarray): (2, M, D) array, lines[0] are starts, lines[1] are ends.
            atol (float): Tolerance for floating point comparison.

        Returns:
            np.ndarray: (N, M) boolean array, True if point i is on line j.
        """
        points = np.atleast_2d(points)  # (N, D)
        line_starts = np.atleast_2d(lines[0])  # (M, D)
        line_ends = np.atleast_2d(lines[1])  # (M, D)

        line_vecs = line_ends - line_starts  # (M, D)
        point_vecs = points[:, None, :] - line_starts[None, :, :]  # (N, M, D)

        line_lens = np.linalg.norm(line_vecs, axis=1)  # (M,)
        line_lens = np.where(line_lens == 0, 1, line_lens)  # Avoid division by zero

        t = np.einsum("nmd,md->nm", point_vecs, line_vecs) / (line_lens**2)  # type: ignore

        on_segment = (t >= -atol) & (t <= 1 + atol)

        closest = line_starts[None, :, :] + t[:, :, None] * line_vecs[None, :, :]

        dist = np.linalg.norm(points[:, None, :] - closest, axis=2)
        is_on_line = np.isclose(dist, 0, atol=atol)

        return on_segment & is_on_line, t

    def _get_edges_between_vertices(
        self, vertices: list[ngs.NodeId], mesh: ngs.Mesh
    ) -> list[ngs.NodeId]:
        """
        Given a set of vertices on a corresponding mesh, find all edges that connect them.
        Args:
            vertices (list[ngs.NodeId]): List of vertex IDs to find edges between.
            mesh (ngs.Mesh): The mesh containing the vertices and edges.
        """
        num_vertices = len(vertices)
        edges = []
        for i in range(num_vertices - 1):
            v1, v2 = vertices[i], vertices[i + 1]
            edges1 = set(mesh[v1].edges)
            edges2 = set(mesh[v2].edges)
            edges += list(edges1.intersection(edges2))
        return edges

    def _refine_mesh(self, mesh: ngs.Mesh):
        """
        Refine the (coarse) mesh in-place to create a fine mesh that can be used for the fespace construction
        and obtaining the prolongation operator
        """
        coarse_mesh = copy.deepcopy(mesh)
        for _ in range(self.refinement_levels):
            mesh.Refine()
        return mesh, coarse_mesh

    ########################
    # domain decomposition #
    ########################
    def get_subdomains(self):
        """
        Identify and return the mapping of coarse mesh elements to their corresponding fine mesh domains.

        Returns:
            dict: Mapping of coarse elements to their fine mesh interior elements and edges.
        """
        subdomains = {}
        coarse_elements = self.coarse_mesh.Elements()
        num_coarse_elements = self.coarse_mesh.ne
        num_elements_per_coarse_element = (
            4**self.refinement_levels
        )  # NOTE: True for any 2D object, see https://www.youtube.com/watch?v=FnRhnZbDprE

        # coarse elements are taken to be subdomains
        task = self.progress.add_task(
            "Calculating subdomains",
            total=self.coarse_mesh.ne,
        )
        for i, subdomain in enumerate(coarse_elements):

            interior_elements = [
                self.fine_mesh[ngs.ElementId(ngs.VOL, i + j * num_coarse_elements)]
                for j in range(num_elements_per_coarse_element)
            ]
            edges = {}
            for coarse_edge in subdomain.edges:
                edges[coarse_edge] = self.coarse_edges_map[coarse_edge]["fine_edges"]
            subdomains[subdomain] = {
                "interior": interior_elements,
                "edges": edges,
            }

            self.progress.advance(task)
        task = self.progress.remove_task(task)
        LOGGER.debug("calculated subdomains")
        return subdomains

    def extend_subdomains(self):
        """
        Extend the subdomains by adding layers of fine mesh elements.

        This method iteratively adds layers of fine mesh elements to each subdomain based on the
        vertices on the edges of the subdomain and the interior elements.
        """
        layer_task = self.progress.add_task(
            f"[cyan]Extending subdomains with {self.layers} layers",
            total=self.layers,
        )
        for layer_idx in range(1, self.layers + 1):
            self._extend_subdomains(layer_idx)
            self.progress.advance(layer_task)

        # remove the layer task and log the completion
        self.progress.remove_task(layer_task)
        LOGGER.debug(f"extended subdomains with {self.layers} layers")

    def _extend_subdomains(self, layer_idx: int):
        """
        Extend the subdomains by adding layers of fine mesh elements.

        Args:
            layer_idx (int, optional): The new layer's index. Defaults to 1.
        """
        subdomains_task = self.progress.add_task(
            f"[green]Extending subdomains (layer {layer_idx})",
            total=len(self.subdomains),
        )
        for subdomain, subdomain_data in self.subdomains.items():
            # get all interior elements of the subdomain
            interior_elements = [el.nr for el in subdomain_data["interior"]]
            for prev_layer_idx in range(1, layer_idx):
                interior_elements += [
                    el.nr for el in subdomain_data[f"layer_{prev_layer_idx}"]
                ]

            # get all vertices on the edge of the domain
            edge_vertices = []
            if layer_idx == 1:  # we use coarse edge map for the first layer
                edge_vertices.extend(subdomain.vertices)
                for coarse_edge in subdomain_data["edges"].keys():
                    edge_vertices.extend(
                        list(self.coarse_edges_map[coarse_edge]["fine_vertices"])
                    )
            if (
                layer_idx > 1
            ):  # we use vertices of the previous layer elements for subsequent layers
                layer_elements = subdomain_data[f"layer_{layer_idx - 1}"]
                for el in layer_elements:
                    mesh_el = self.fine_mesh[el]
                    edge_vertices.extend(mesh_el.vertices)

            # get all elements associated with the edge vertices
            layer_elements = []
            for edge_vertex in edge_vertices:
                mesh_vertex = self.fine_mesh[edge_vertex]
                layer_elements.extend([el.nr for el in mesh_vertex.elements])

            # remove duplicates
            layer_elements = set(layer_elements)

            # convert to numpy array for easier manipulation
            layer_elements = np.array(list(layer_elements))

            # make unique
            layer_elements = np.unique(layer_elements)

            # filter out elements that are already in the interior or previous layers
            mask = np.isin(
                layer_elements, interior_elements, assume_unique=True, invert=True
            )
            layer_elements = layer_elements[mask]

            # add the new layer elements to the subdomain data
            subdomain_data[f"layer_{layer_idx}"] = [
                self.fine_mesh[ngs.ElementId(el)] for el in layer_elements
            ]

            self.progress.advance(subdomains_task)

        # remove the subdomains task and log the completion
        self.progress.remove_task(subdomains_task)
        LOGGER.debug(f"extended subdomains with layer {layer_idx}/{self.layers}")

    ###########################
    # interface decomposition #
    ###########################
    def get_connected_components(self) -> dict:
        """
        Finds all the connected components in the fine mesh based on the coarse mesh subdomains.

        That is,
        1) each coarse node,
        2) for each subdomain edge all fine edges and fine vertices that belong to it (except for those vertices belonging to 1)
        3) for each subdomain face all fine faces, fine edges, fine vertices that belong to it (except for those belonging to 1 & 2)

        Returns:
            dict: A dictionary mapping each coarse mesh element to its fine mesh elements.
        """
        connected_components = {}

        connected_components["coarse_nodes"] = list(self.coarse_mesh.vertices)

        connected_components["edges"] = {}
        for coarse_edge, nodes in self.coarse_edges_map.items():
            fine_edges = nodes["fine_edges"]
            fine_vertices = nodes["fine_vertices"]
            connected_components["edges"][coarse_edge] = fine_edges + fine_vertices

        # NOTE: this code is only necessary for 3D meshes
        connected_components["faces"] = {}

        LOGGER.debug("calculated connected components")
        return connected_components

    def get_connected_component_tree(self, connected_components: dict) -> dict:
        """
        Finds the tree of connected components in the fine mesh based on the coarse mesh subdomains.

        Note a 2D mesh does not have faces so the fine edge lists will be empty.
        Returns:
            dict: A dictionary with the following structure:
                {
                    coarse_node_i1: {
                        coarse_edge_j1: [fine_edge_1, fine_edge_2, ..., fine_vertex_1, fine_vertex_2, ...],
                        coarse_edge_j2:{...},
                        ...,
                        coarse_edge_jm:{...},
                        coarse_face_k1: [fine_face_1, fine_face_2, ..., fine_edge_1, fine_edge_2, ..., fine_vertex_1, fine_vertex_2, ...],
                        coarse_face_k2: {...},
                        ...,
                        coarse_face_kn: {...}
                    }
                    coarse_node_i2: {...},
                    ...,
                    coarse_node_in: {...}
                }
        """
        component_tree = {
            coarse_node: {
                "edges": {
                    self.coarse_mesh[coarse_edge]: connected_components["edges"][
                        coarse_edge
                    ]
                    for coarse_edge in coarse_node.edges
                },
                # "faces": { #NOTE: this is only for 3D meshes
                #     self.coarse_mesh[coarse_face]: connected_components["faces"][
                #         coarse_face
                #     ]
                #     for coarse_face in coarse_node.faces
                # },
            }
            for coarse_node in connected_components["coarse_nodes"]
        }
        LOGGER.debug("calculated connected component tree")
        return component_tree

    #######################
    # generate edge slabs #
    #######################
    def generate_edge_slabs(self) -> dict:
        """
        Generates the mapping from coarse edges to fine mesh elements that are
        in slabs that are extruded from the coarse edge vertices.

        Requires a refinement level of at least 4

        These slabs can be used to create a coefficient function on the fine mesh that has high
        contrast on the slab elements and low contrast on the rest of the mesh. These
        slabs should be larger than the domain overlap, in order to create edge inclusions.
        Hence the number of slab layers is equal to the number of layers in the TwoLevelMesh plus one.

        This functions can generates two sets of edge inclusions: single and double slabs.
        """
        # prevent creation of slabs if refinement level is too low
        if self.refinement_levels < 4:
            LOGGER.warning(
                "Refinement level is too low for edge slab generation. "
                "Please set refinement_levels >= 4."
            )
            return {}

        # main output dictionary
        edge_slabs = {}

        # get num vertices on subdomain edges (without coarse nodes)
        num_edge_vertices = 2**self.refinement_levels - 2

        # find index center vertex on coarse edge
        center_idx = num_edge_vertices // 2

        # number of layers in the slab (slab height ~ half the subdomain diameter)
        slab_layers = center_idx // 2

        task = self.progress.add_task(
            "[cyan]Generating edge slabs",
            total=len(self.coarse_edges_map) + len(self.coarse_mesh.vertices),
        )
        for coarse_edge in self.coarse_edges_map.keys():
            single_slab_elements = self.get_slab_elements(
                coarse_edge,
                [center_idx],
                slab_layers=slab_layers,
            )
            double_slab_elements = self.get_slab_elements(
                coarse_edge,
                [center_idx - 2, center_idx + 2],
                slab_layers=slab_layers,
            )

            # store the slab elements in the output dictionary
            edge_slabs[coarse_edge.nr] = {
                "single": single_slab_elements,
                "double": double_slab_elements,
            }
            self.progress.advance(task)

        # edge slabs around coarse nodes
        idx = num_edge_vertices // 4
        edge_slabs["around_coarse_nodes"] = {}
        coarse_edges_processed = []
        for coarse_node in self.coarse_mesh.vertices:
            edge_slabs["around_coarse_nodes"][coarse_node.nr] = []
            for coarse_edge in self.coarse_mesh[coarse_node].edges:
                _idx = idx
                if np.isin(coarse_edge.nr, coarse_edges_processed):
                    # invert index if coarse edge was already processed
                    # NOTE: this prevents getting overlapping slabs
                    _idx = num_edge_vertices - _idx
                slabs = self.get_slab_elements(
                    coarse_edge,
                    [_idx],
                    slab_layers=slab_layers,
                )
                edge_slabs["around_coarse_nodes"][coarse_node.nr].extend(*slabs)
                coarse_edges_processed.append(coarse_edge.nr)

            self.progress.advance(task)

        # remove the task and log the completion
        self.progress.remove_task(task)
        LOGGER.debug("generated edge slabs")

        return edge_slabs

    def get_slab_elements(
        self, coarse_edge: ngs.NodeId, idxs: list[int], slab_layers: int
    ) -> list[list[ngs.ElementId]]:
        """
        Get the slab elements for a given coarse edge centered at the idxs given in idxs.

        The allowed indices in idxs are 0,1, ..., n_fv - 1, where n_vf = 2**self.refinement_levels - 1 is
        the number of fine vertices on the coarse edge (so excluding the coarse nodes).

        The values in idx should correspond to the fine vertices on the coarse edge.
        The width of a slab will always be 2 fine edges or 3 fine vertices, the
        middle of which will correspond to the index(-ices) in idxs.

        The number of layers in the slab can be specified by slab_layers, which is the number of fine edges the
        slab will extend in both extrusion directions. This height is still limited though by the diameter of the subdomain
        into which the slab is extruded, so the slab will not extend beyond the subdomain boundary.
        """
        # instantiate output
        slabs = []

        # get the center vertex and the edge corners
        fine_vertices = np.array(self.coarse_edges_map[coarse_edge]["fine_vertices"])

        # get main edge direction
        v0, v1 = self.fine_mesh[coarse_edge].vertices
        p0, p1 = (
            self.fine_mesh[v0].point,
            self.fine_mesh[v1].point,
        )
        main_edge_direction = np.array(p1) - np.array(p0)
        main_edge_direction /= np.linalg.norm(main_edge_direction)

        # skip edge if current mesh is quad and direction parrallel to the y axis
        if self.quad and np.isclose(main_edge_direction[0], 0):
            return [[] for _ in range(len(idxs))]

        # main loop over the indices in idxs
        for idx in idxs:
            center_vertex = fine_vertices[idx]
            edge_corners = fine_vertices[[idx - 1, idx + 1]].tolist()

            # get extrusion directions
            upper_direction, lower_direction = self.get_extrusion_directions(
                center_vertex, main_edge_direction
            )

            # only consider existing extrusion directions
            directions = []
            if upper_direction.size > 0:
                directions.append(upper_direction)
            if lower_direction.size > 0:
                directions.append(lower_direction)

            # loop over both extrusion directions to find slab elements
            slab_elements = []
            for direction in directions:
                # extrude the edge corners (to get parallelogram corners)
                interior_corners, all_outer_vertices = self.extrude_vertices(
                    vertices=edge_corners,
                    extrusion_direction=direction,
                    num_extrusions=slab_layers,
                )

                # extrude the center vertex (to get all inner vertices)
                _, all_inner_vertices = self.extrude_vertices(
                    vertices=[center_vertex],
                    extrusion_direction=direction,
                    num_extrusions=slab_layers,
                )

                # get corner of the parallelogram
                corners = edge_corners + interior_corners
                corner_points = np.array([self.fine_mesh[v].point for v in corners])

                # get all elements in and near the parallelogram
                el_nrs = []
                el_points = []
                # we loop over all available vertices to get a full covering of the parallelogram
                for v in all_outer_vertices + all_inner_vertices:
                    mesh_el = self.fine_mesh[v]
                    for el in mesh_el.elements:
                        el_nrs.append(el.nr)
                        points = []
                        for vertex in self.fine_mesh[el].vertices:
                            points.append(self.fine_mesh[vertex].point)
                        el_points.append(np.array(points))
                el_nrs = np.array(el_nrs)
                el_points = np.array(el_points)

                # perform parallellogram check
                inside_elements = self.points_in_parallelogram(
                    corner_points,
                    el_points,
                )

                # pick only elements that have all points inside the parallelogram
                inside_elements = np.all(inside_elements, axis=1)

                # get indices of elements that are inside the parallelogram
                inside_element_indices = el_nrs[inside_elements].tolist()

                # store the slab elements
                slab_elements.extend(inside_element_indices)

            # store the slab elements in the output list
            slabs.append(slab_elements)

        return slabs

    def get_extrusion_directions(
        self,
        center_vertex: ngs.NodeId,
        main_edge_direction: np.ndarray,
    ):
        upper_edges = []
        upper_edge_directions = []
        lower_edges = []
        lower_edge_directions = []
        perpendicular_edge_direction = np.array(
            [-main_edge_direction[1], main_edge_direction[0]]
        )
        for edge in self.fine_mesh[center_vertex].edges:
            v0, v1 = self.fine_mesh[edge].vertices
            p0, p1 = (
                self.fine_mesh[v0].point,
                self.fine_mesh[v1].point,
            )
            edge_direction = np.array(p1) - np.array(p0)
            edge_direction /= np.linalg.norm(edge_direction)
            inner_product = np.dot(edge_direction, perpendicular_edge_direction)
            if np.isclose(np.abs(inner_product), 0.0):
                continue  # edge is parallel to the main edge direction
            elif inner_product > 0:
                upper_edges.append(edge)
                upper_edge_directions.append(edge_direction)
            else:
                lower_edges.append(edge)
                lower_edge_directions.append(edge_direction)

        # get extrusion directions
        upper_extrusion_direction = np.array([])
        if len(upper_edge_directions) > 1:
            upper_extrusion_direction = self._get_middle_direction(
                upper_edge_directions, main_edge_direction
            )
        elif len(upper_edge_directions) == 1:
            upper_extrusion_direction = upper_edge_directions[0]

        lower_extrusion_direction = np.array([])
        if len(lower_edge_directions) > 1:
            lower_extrusion_direction = self._get_middle_direction(
                lower_edge_directions, main_edge_direction
            )
        elif len(lower_edge_directions) == 1:
            lower_extrusion_direction = lower_edge_directions[0]

        return upper_extrusion_direction, lower_extrusion_direction

    def _get_middle_direction(
        self, directions: list[np.ndarray], main_edge_direction: np.ndarray
    ):
        angles = []
        for direction in directions:
            dot = np.dot(direction, main_edge_direction)
            det = (
                direction[0] * main_edge_direction[1]
                - direction[1] * main_edge_direction[0]
            )  # 2D cross product
            angle = np.arctan2(det, dot)
            angles.append(angle)
        angles = np.array(angles)
        angles_sorted = np.argsort(angles)
        middle_index = len(angles_sorted) // 2
        middle_direction = directions[angles_sorted[middle_index]]
        return middle_direction

    def extrude_vertices(
        self,
        vertices: list[ngs.NodeId],
        extrusion_direction: np.ndarray,
        num_extrusions: int = 1,
    ):
        """
        Extrude a list of vertices along a given direction.

        Args:
            vertices (list[ngs.NodeId]): List of vertices to extrude.
            extrusion_direction (np.ndarray): Direction vector for extrusion.
            num_extrusions (int): Number of times to extrude the edge.

        Returns:
        """
        all_vertices = vertices.copy()
        outer_vertices = vertices.copy()
        for _ in range(num_extrusions):
            outer_edges = []
            for v in outer_vertices:
                outer_edges.extend(self.fine_mesh[v].edges)

            new_extrusion_edges = []
            for edge in outer_edges:
                # get outward orientation of the edge
                start_point = None
                edge_end_point = None
                v0, v1 = self.fine_mesh[edge].vertices
                if v0 in outer_vertices:
                    start_point = np.array(self.fine_mesh[v0].point)
                    edge_end_point = np.array(self.fine_mesh[v1].point)
                else:
                    start_point = np.array(self.fine_mesh[v1].point)
                    edge_end_point = np.array(self.fine_mesh[v0].point)

                # check if the edge is parallel to the extrusion direction
                edge_direction = edge_end_point - start_point
                edge_direction /= np.linalg.norm(edge_direction)
                inner_product = np.dot(edge_direction, extrusion_direction)
                if np.isclose(inner_product, 1.0):
                    new_extrusion_edges.append(edge)

            # update outer vertices
            new_outer_vertices = []
            for edge in new_extrusion_edges:
                v0, v1 = self.fine_mesh[edge].vertices
                if v0 not in all_vertices:
                    new_outer_vertices.append(v0)
                if v1 not in all_vertices:
                    new_outer_vertices.append(v1)
            all_vertices.extend(new_outer_vertices)
            outer_vertices = new_outer_vertices

        return outer_vertices, all_vertices

    def points_in_parallelogram(
        self, corners: np.ndarray, points_to_check: np.ndarray, tol: float = 1e-9
    ) -> np.ndarray:
        # 1. Select corners[0] as point A (origin for parallelogram vectors).
        A = corners[0, :]

        # 2. Determine the two edge vectors vec1 and vec2 originating from A.
        v_AP1 = corners[1, :] - A
        v_AP2 = corners[2, :] - A
        v_AP3 = corners[3, :] - A

        vec1 = None
        vec2 = None

        # Check combinations to find which two sum to the third.
        if np.allclose(v_AP1 + v_AP2, v_AP3, atol=tol):
            vec1 = v_AP1
            vec2 = v_AP2
        elif np.allclose(v_AP1 + v_AP3, v_AP2, atol=tol):
            vec1 = v_AP1
            vec2 = v_AP3
        elif np.allclose(v_AP2 + v_AP3, v_AP1, atol=tol):
            vec1 = v_AP2
            vec2 = v_AP3
        else:
            raise ValueError(
                "The provided corners do not appear to form a parallelogram from which "
            )

        # Reshape points_to_check to (-1, 2) for vectorized computation
        orig_shape = points_to_check.shape[:-1]
        AP_check = points_to_check.reshape(-1, 2) - A  # (N*M, 2)

        # 4. Solve AP_check = u * vec1 + v * vec2 for u and v.
        det_M = vec1[0] * vec2[1] - vec1[1] * vec2[0]

        if abs(det_M) < tol:
            raise ValueError(
                "The vectors vec1 and vec2 are linearly dependent or nearly so, "
                "which means the parallelogram is degenerate."
            )

        # Numerator for u: cross_product_2d(AP_check, vec2)
        u_numerator = AP_check[:, 0] * vec2[1] - AP_check[:, 1] * vec2[0]  # Shape (N,)

        # Numerator for v: cross_product_2d(vec1, AP_check)
        v_numerator = vec1[0] * AP_check[:, 1] - vec1[1] * AP_check[:, 0]  # Shape (N,)

        u = u_numerator / det_M
        v = v_numerator / det_M

        # 5. A point is inside or on the boundary if 0 <= u <= 1 and 0 <= v <= 1.
        is_inside_u = (u >= -tol) & (u <= 1 + tol)
        is_inside_v = (v >= -tol) & (v <= 1 + tol)

        return (is_inside_u & is_inside_v).reshape(orig_shape)

    ####################
    # saving & loading #
    ####################
    def save(self, save_vtk_meshes: bool = False):
        """
        Save the mesh, metadata, and subdomain information to disk.

        Args:
            file_name (str, optional): Name of the file to save the mesh and metadata. Defaults to "".
        """
        self.progress = PROGRESS.get_active_progress_bar(self.progress)
        task = self.progress.add_task(
            f"Saving TwoLevelMesh",
            total=5 if save_vtk_meshes else 4,
        )

        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            f"Saving TwoLevelMesh 1/H = {1/self.coarse_mesh_size:.0f} to %s",
            self.save_dir,
        )

        # mesh and metadata
        self._save_metadata()
        self.progress.advance(task)

        # meshes (vtk)
        if save_vtk_meshes:
            self._save_vtk_meshes()
            self.progress.advance(task)

        # coarse edges map
        self._save_coarse_edges_map()
        self.progress.advance(task)

        # overlap layers
        self._save_subdomain_layers()
        self.progress.advance(task)

        # edge slabs for edge inclusions
        self._save_edge_slabs()
        self.progress.advance(task)

        LOGGER.info(f"Saved TwoLevelMesh")
        self.progress.soft_stop()

    def _save_metadata(self):
        """
        Save mesh and domain metadata to a JSON file.
        """
        metadata = {
            "lx": self.lx,
            "ly": self.ly,
            "coarse_mesh_size": self.coarse_mesh_size,
            "refinement_levels": self.refinement_levels,
            "layers": self.layers,
            "quad": self.quad,
        }
        metadata_path = self.save_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        LOGGER.debug("saved metadata")

    def _save_vtk_meshes(self):
        """
        Saves the fine and coarse mesh data to disk in .vtk formats.

        Args:
            save_vtk (bool, optional): If True, saves the meshes in VTK format as well. Defaults to True.
        """
        vtk = ngs.VTKOutput(
            self.fine_mesh,
            coefs=[],
            names=[],
            filename=str(self.save_dir / "fine_mesh"),
        )
        vtk.Do()
        vtk = ngs.VTKOutput(
            self.coarse_mesh,
            coefs=[],
            names=[],
            filename=str(self.save_dir / "coarse_mesh"),
        )
        vtk.Do()

        LOGGER.debug("saved fine and coarse meshes in .vtk formats")

    def _save_coarse_edges_map(self):
        """
        Save the mapping of fine edges to coarse edges.
        """
        coarse_edges_map_path = self.save_dir / "coarse_edges_map.json"
        with open(coarse_edges_map_path, "w") as f:
            coarse_edges_map = {
                coarse_edge.nr: {
                    "fine_vertices": [
                        vertex.nr
                        for vertex in self.coarse_edges_map[coarse_edge][
                            "fine_vertices"
                        ]
                    ],
                    "fine_edges": [
                        edge.nr
                        for edge in self.coarse_edges_map[coarse_edge]["fine_edges"]
                    ],
                }
                for coarse_edge in self.coarse_mesh.edges
            }
            json.dump(coarse_edges_map, f, indent=4)

        LOGGER.debug("saved coarse edges map")

    def _save_subdomain_layers(self):
        """
        Save the subdomain layers to a JSON file.
        """
        subdomains_path = self.save_dir / "subdomains_layers.json"
        with open(subdomains_path, "w") as f:
            subdomains_data = {
                subdomain.nr: {
                    **{
                        f"layer_{layer_idx}": [
                            el.nr for el in subdomain_data[f"layer_{layer_idx}"]
                        ]
                        for layer_idx in range(1, self.layers + 1)
                    },
                }
                for subdomain, subdomain_data in self.subdomains.items()
            }
            json.dump(subdomains_data, f, indent=4)

        LOGGER.debug("saved subdomain layers")

    def _save_edge_slabs(self):
        """
        Save the edge slabs to a JSON file.
        """
        edge_slabs_path = self.save_dir / "edge_slabs.json"
        with open(edge_slabs_path, "w") as f:
            json.dump(self.edge_slabs, f, indent=4)
        LOGGER.debug("saved edge slabs")

    @classmethod
    def load(
        cls,
        mesh_params: MeshParams,
        progress: Optional[PROGRESS] = None,
    ):
        """
        Load the mesh and metadata from files and return a TwoLevelMesh instance.

        Args:
            file_name (str, optional): Name of the file to load the mesh and metadata. Defaults to "".

        Returns:
            TwoLevelMesh: Loaded TwoLevelMesh instance.
        """
        fp = cls.get_save_dir(mesh_params)
        if fp.exists():
            progress = PROGRESS.get_active_progress_bar(progress)
            obj = cls.__new__(cls)
            setattr(obj, "progress", progress)
            task = obj.progress.add_task(
                "Loading TwoLevelMesh",
                total=5,
            )
            LOGGER.info("Loading TwoLevelMesh from %s", fp)

            # metadata
            obj._load_metadata(fp)
            obj.progress.update(
                task,
                advance=1,
                description=obj.progress.get_description(task)
                + f" 1/H = {1/obj.coarse_mesh_size:.0f}",
            )

            # meshes
            obj._load_meshes(fp)
            obj.progress.advance(task)

            # coarse edges map
            obj._load_coarse_edges_map(fp)
            obj.progress.advance(task)

            # subdomains
            setattr(obj, "subdomains", obj.get_subdomains())
            obj.progress.advance(task)

            # overlap layers
            obj._load_subdomain_layers(fp)
            obj.progress.advance(task)

            # connected components
            setattr(obj, "connected_components", obj.get_connected_components())
            obj.progress.advance(task)

            # connected component tree
            setattr(
                obj,
                "connected_component_tree",
                obj.get_connected_component_tree(obj.connected_components),
            )
            obj.progress.advance(task)

            # edge slabs for edge inclusions
            obj.edge_slabs = obj._load_edge_slabs(fp)
            obj.progress.advance(task)

            LOGGER.info(
                f"Finished loading TwoLevelMesh 1/H = {1/obj.coarse_mesh_size:.0f}"
            )
            LOGGER.debug(str(obj))
            obj.progress.soft_stop()
        else:
            raise FileNotFoundError("Metadata file %s does not exist." % str(fp))
        return obj

    def _load_metadata(self, fp: Path):
        """
        Load mesh and domain metadata from a JSON file.
        """
        metadata_path = fp / "metadata.json"
        if not metadata_path.exists():
            msg = "Metadata file %s does not exist"
            LOGGER.error(msg, metadata_path)
            raise FileNotFoundError(msg % str(metadata_path))
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        self.lx = metadata["lx"]
        self.ly = metadata["ly"]
        self.coarse_mesh_size = metadata["coarse_mesh_size"]
        self.refinement_levels = metadata["refinement_levels"]
        self.layers = metadata["layers"]
        self.quad = metadata["quad"]
        self.mesh_params = MeshParams(
            lx=self.lx,
            ly=self.ly,
            Nc=int(1 / self.coarse_mesh_size),
            refinement_levels=self.refinement_levels,
            layers=self.layers,
            quad=self.quad,
        )
        LOGGER.debug(f"loaded metadata")

    def _load_meshes(self, fp: Path):
        """
        Load fine and coarse meshes from disk.
        """
        mesh = self.create_mesh()
        self.fine_mesh, self.coarse_mesh = self._refine_mesh(mesh)
        LOGGER.debug(f"regenerated meshes")

    def _load_coarse_edges_map(self, fp: Path):
        """
        Load the mapping of fine edges to coarse edges from a JSON file.
        """
        coarse_edges_map_path = fp / "coarse_edges_map.json"
        if not coarse_edges_map_path.exists():
            msg = "Coarse edges map file %s does not exist. "
            LOGGER.error(msg, coarse_edges_map_path)
            raise FileNotFoundError(msg % str(coarse_edges_map_path))
        with open(coarse_edges_map_path, "r") as f:
            coarse_edges_map = json.load(f)
        self.coarse_edges_map = {
            self.coarse_mesh.edges[int(coarse_edge_nr)]: {
                "fine_vertices": [
                    self.fine_mesh[self.fine_mesh.vertices[v]]
                    for v in data["fine_vertices"]
                ],
                "fine_edges": [
                    self.fine_mesh[self.fine_mesh.edges[e]] for e in data["fine_edges"]
                ],
            }
            for coarse_edge_nr, data in coarse_edges_map.items()
        }
        LOGGER.debug(f"loaded coarse edges map")

    def _load_subdomain_layers(self, fp: Path):
        """
        Load the subdomain layers from a JSON file.
        """
        subdomains_path = fp / "subdomains_layers.json"
        if not subdomains_path.exists():
            msg = "Subdomains layers file %s does not exist. "
            LOGGER.error(msg, subdomains_path)
            raise FileNotFoundError(msg % str(subdomains_path))
        with open(subdomains_path, "r") as f:
            subdomains_data = json.load(f)

        for subdomain in self.subdomains.keys():
            subdomain_data = subdomains_data[str(subdomain.nr)]
            for layer_idx in range(1, self.layers + 1):
                layer_elements = [
                    self.fine_mesh[ngs.ElementId(el)]
                    for el in subdomain_data[f"layer_{layer_idx}"]
                ]
                self.subdomains[subdomain][f"layer_{layer_idx}"] = layer_elements
        LOGGER.debug(f"recovered {self.layers} overlap layers for each subdomain")

    def _load_edge_slabs(self, fp: Path) -> dict:
        """
        Load the edge slabs from a JSON file.

        Args:
            fp (Path): Path to the folder containing the edge slabs file.

        Returns:
            dict: The edge slabs mapping.
        """
        edge_slabs_path = fp / "edge_slabs.json"
        if not edge_slabs_path.exists():
            msg = "Edge slabs file %s does not exist. "
            LOGGER.error(msg, edge_slabs_path)
            raise FileNotFoundError(msg % str(edge_slabs_path))
        elif self.refinement_levels < 4:
            LOGGER.warning("No edge slabs generated for refinement level < 4.")
            return {}
        with open(edge_slabs_path, "r") as f:
            _edge_slabs = json.load(f)
        # Convert string keys back to int
        edge_slabs = {}
        for k, v in _edge_slabs.items():
            if k == "around_coarse_nodes":
                edge_slabs[k] = {int(k2): v2 for k2, v2 in v.items()}
            else:
                edge_slabs[int(k)] = v
        LOGGER.debug(f"loaded edge slabs")
        return edge_slabs

    @property
    def save_dir(self):
        """
        Directory where the mesh and subdomain data are saved.
        """
        return self.get_save_dir(self.mesh_params)

    @classmethod
    def get_save_dir(cls, mesh_params: MeshParams) -> Path:
        """
        Get the save directory based on mesh parameters.

        Args:
            mesh_params (MeshParams): Parameters for the mesh.

        Returns:
            Path: The save directory path.
        """
        folder_name = cls.SAVE_STRING.format(
            "quad" if mesh_params.quad else "tri",
            mesh_params.lx,
            mesh_params.ly,
            1 / mesh_params.coarse_mesh_size,
            mesh_params.refinement_levels,
            mesh_params.layers,
        )
        return DATA_DIR / folder_name

    @save_dir.setter
    def save_dir(self, folder_name: str):
        """
        Set the folder where the mesh and subdomain data are saved.

        Args:
            folder_name (str): Name of the folder.
        """
        self._save_folder = DATA_DIR / folder_name
        if not self._save_folder.exists():
            self._save_folder.mkdir(parents=True, exist_ok=True)

    ####################
    # meta info string #
    ####################
    def __str__(self):
        return (
            f"Fine mesh:"
            f"\n\telements: {self.fine_mesh.ne}"
            f"\n\tvertices: {self.fine_mesh.nv}"
            f"\n\tedges: {len(self.fine_mesh.edges)}"
            f"\nCoarse mesh:"
            f"\n\telements: {self.coarse_mesh.ne}"
            f"\n\tvertices: {self.coarse_mesh.nv}"
            f"\n\tedges: {len(self.coarse_mesh.edges)}"
        )

    ############
    # plotting #
    ############
    def plot_mesh(self, ax: Axes, mesh_type: str = "fine"):
        """
        Plot the fine or coarse mesh using matplotlib.

        Args:
            mesh_type (str, optional): Type of mesh to plot ('fine' or 'coarse'). Defaults to 'fine'.

        Returns:
            tuple: (figure, ax) Matplotlib figure and axis.
        """
        if mesh_type == "fine":
            mesh = self.fine_mesh
            linewidth = 1.0
            alpha = 1.0
            fillcolor = "lightgray"
            edgecolor = "black"
        elif mesh_type == "coarse":
            mesh = self.coarse_mesh
            linewidth = 2.0
            alpha = 0.5
            fillcolor = "lightblue"
            edgecolor = "darkblue"
        else:
            msg = f"Invalid mesh_type '{mesh_type}'. " "Please use 'fine' or 'coarse'."
            LOGGER.error(msg)
            raise ValueError(msg)

        for el in mesh.Elements():
            self.plot_element(
                ax,
                el,
                mesh,
                fillcolor=fillcolor,
                edgecolor=edgecolor,
                alpha=alpha,
                linewidth=linewidth,
            )
        return ax

    def plot_domains(
        self,
        ax: Axes,
        domains: list | int = 1,
        plot_layers: bool = False,
        opacity: float = 0.9,
        fade_factor: float = 1.5,
    ):
        """
        Visualize the subdomains and optional layers using matplotlib.

        Args:
            domains (list or int, optional): Which domains to plot. Defaults to 1.
            plot_layers (bool, optional): Whether to plot layers. Defaults to False.
            opacity (float, optional): Opacity of the domain fill. Defaults to 0.9.
            fade_factor (float, optional): Controls fading of layers. Defaults to 1.5.

        Returns:
            tuple: (figure, ax) Matplotlib figure and axis.
        """
        domains_int_toggle = isinstance(domains, int)
        domains_list_toggle = isinstance(domains, list)
        for i, (subdomain, subdomain_data) in enumerate(self.subdomains.items()):
            # plot coarse mesh element without fill and thick border
            self.plot_element(
                ax,
                subdomain,
                self.coarse_mesh,
                fillcolor=None,
                edgecolor="black",
                alpha=1.0,
                linewidth=2.0,
                zorder=TwoLevelMesh.ZORDERS["edges"] + 0.1,
            )
            domain_elements = subdomain_data["interior"]
            if domains_int_toggle:
                if i % domains != 0:
                    continue
            elif domains_list_toggle:
                if subdomain.nr not in domains:
                    continue
            fillcolor = next(self.subdomain_colors)
            for domain_el in domain_elements:
                self.plot_element(
                    ax,
                    domain_el,
                    self.fine_mesh,
                    fillcolor=fillcolor,
                    alpha=opacity,
                    edgecolor="black",
                )
            if plot_layers:
                for layer_idx in range(1, self.layers + 1):
                    layer_elements = subdomain_data.get(f"layer_{layer_idx}", [])
                    alpha_value = (
                        opacity / (1 + (layer_idx / self.layers) ** fade_factor) / 2
                    )
                    for layer_el in layer_elements:
                        self.plot_element(
                            ax,
                            layer_el,
                            self.fine_mesh,
                            fillcolor="black",
                            alpha=alpha_value,
                            edgecolor="black",
                            label=f"layersLayer {layer_idx} Element",
                            zorder=TwoLevelMesh.ZORDERS["layers"],
                        )
        return ax

    def plot_connected_components(self, ax: Axes):
        """
        Plot the connected components of the fine mesh based on the coarse mesh subdomains.

        Args:
            ax (Axes): Matplotlib axis to plot on.

        Returns:
            Axes: The matplotlib axis with the plotted connected components.
        """
        # coarse nodes (correspond to fine vertices)

        self.plot_vertices(
            ax,
            self.connected_components["coarse_nodes"],
            self.fine_mesh,
            color="red",
            marker="o",
            markersize=20,
        )

        # edges
        for components in self.connected_components["edges"].values():
            self.plot_components(
                ax,
                components,
                self.fine_mesh,
                edge_color="blue",
                vertex_color="green",
                marker="x",
                markersize=15,
                linewidth=1.5,
                linestyle="-",
                zorder=TwoLevelMesh.ZORDERS["edges"] + 0.1,
            )

        # faces
        for coarse_face, components in self.connected_components["faces"].items():
            for face in components:
                raise NotImplementedError(
                    "Plotting connected components for faces is not implemented yet."
                )
        return ax

    def plot_connected_component_tree(self, ax: Axes):
        """
        Plot the connected component tree of the fine mesh based on the coarse mesh subdomains.

        Args:
            ax (Axes): Matplotlib axis to plot on.

        Returns:
            Axes: The matplotlib axis with the plotted connected component tree.
        """
        plotted_edges = set()
        for coarse_node, components in self.connected_component_tree.items():
            color = next(self.subdomain_colors)
            # plot big coarse node
            self.plot_vertices(
                ax,
                [coarse_node],
                self.coarse_mesh,
                color=color,
                marker="x",
                markersize=50,
            )
            for coarse_edge, edge_components in components["edges"].items():
                # plot thick coarse edge showing the connected component
                self.plot_edges(
                    ax,
                    [coarse_edge],
                    self.coarse_mesh,
                    color=color,
                    alpha=0.25,
                    linewidth=8.0,
                    zorder=TwoLevelMesh.ZORDERS["edges"] - 0.1,
                    linestyle="-" if coarse_edge not in plotted_edges else "--",
                )
                plotted_edges.add(coarse_edge)

                # plot fine edges and vertices
                self.plot_components(
                    ax,
                    edge_components,
                    self.fine_mesh,
                    fillcolor=color,
                    edge_color="black",
                    vertex_color="green",
                    marker="x",
                    markersize=15,
                    alpha=0.5,
                    linewidth=1.5,
                    linestyle="-",
                    zorder=TwoLevelMesh.ZORDERS["edges"] + 0.1,
                )

        return ax

    def plot_edge_slabs(self, ax: Axes, which: Literal["single", "double", "vertices"]):
        if which in ["single", "double"]:
            for coarse_edge in self.coarse_edges_map.keys():
                slab_elements = self.edge_slabs[coarse_edge.nr][which]
                for el_nrs in slab_elements:
                    for el_nr in el_nrs:
                        self.plot_element(
                            ax,
                            self.fine_mesh[ngs.ElementId(el_nr)],
                            self.fine_mesh,
                            fillcolor="red",
                            edgecolor="black",
                            alpha=0.5,
                            linewidth=1.0,
                            zorder=TwoLevelMesh.ZORDERS["elements"] + 0.1,
                        )
        elif which == "vertices":
            for slab_elements in self.edge_slabs["around_coarse_nodes"].values():
                for el_nr in slab_elements:
                    self.plot_element(
                        ax,
                        self.fine_mesh[ngs.ElementId(el_nr)],
                        self.fine_mesh,
                        fillcolor="red",
                        edgecolor="black",
                        alpha=0.5,
                        linewidth=1.0,
                        zorder=TwoLevelMesh.ZORDERS["elements"] + 0.1,
                    )
        else:
            msg = "Invalid 'which' parameter. Use 'single', 'double', or 'vertices'"
            LOGGER.error(msg)
            raise ValueError(msg)

    @staticmethod
    def plot_element(
        ax: Axes,
        element,
        mesh,
        **kwargs,
    ):
        """
        Plot a single mesh element on a matplotlib axis.

        Args:
            ax (Axes): Matplotlib axis to plot on.
            element: ElementId of the element to plot.
            mesh: NGSolve mesh containing the element.
            fillcolor (str, optional): Color for the element. Defaults to "blue".
            edgecolor (str, optional): Color for the element edges. Defaults to "black".
            alpha (float, optional): Opacity of the fill. Defaults to 1.0.
            label (str, optional): Label for the element (optional).
        """
        vertices = [mesh[v].point for v in mesh[element].vertices]
        polygon = Polygon(
            vertices,
            closed=True,
            fill=True if kwargs.get("fillcolor") else False,
            facecolor=kwargs.get("fillcolor", "blue"),
            edgecolor=kwargs.get("edgecolor", "black"),
            label=kwargs.get("label"),
            alpha=kwargs.get("alpha", 1.0),
            linewidth=kwargs.get("linewidth", 1.5),
            zorder=kwargs.get("zorder", TwoLevelMesh.ZORDERS["elements"]),
        )
        ax.add_patch(polygon)

    @staticmethod
    def plot_components(
        ax: Axes,
        components: list[ngs.NodeId],  # list of vertices, edges, and/or faces
        mesh,
        **kwargs,
    ):
        vertices = []
        edges = []
        faces = []
        for component in components:
            mesh_comp = mesh[component]
            try:  # edge component
                _ = mesh_comp.vertices
                try:
                    _ = mesh_comp.edges
                except TypeError:
                    edges.append(mesh_comp)
                    continue
            except TypeError:  # vertex component
                vertices.append(mesh_comp)
                continue
            faces.append(mesh_comp)  # face component

        TwoLevelMesh.plot_vertices(
            ax, vertices, mesh, color=kwargs.get("vertex_color", "green"), **kwargs
        )
        TwoLevelMesh.plot_edges(
            ax,
            edges,
            mesh,
            color=kwargs.get("edge_color", "black"),
            **kwargs,
        )
        TwoLevelMesh.plot_faces(
            ax, faces, mesh, color=kwargs.get("face_color", "yellow"), **kwargs
        )

    @staticmethod
    def plot_vertices(
        ax: Axes,
        vertices: list,  # list of vertices
        mesh,
        **kwargs,
    ):
        """
        Plot a single mesh node on a matplotlib axis.

        Args:
            ax (Axes): Matplotlib axis to plot on.
            node: NodeId of the node to plot.
            mesh: NGSolve mesh containing the node.
            color (str, optional): Color for the node. Defaults to "red".
            marker (str, optional): Marker style for the node. Defaults to "o".
            markersize (int, optional): Size of the marker. Defaults to 5.
            label (str, optional): Label for the node (optional).
        """
        coarse_node_coords = np.array([mesh.vertices[v.nr].point for v in vertices])
        ax.scatter(
            coarse_node_coords[:, 0],
            coarse_node_coords[:, 1],
            c=kwargs.get("color", "red"),
            marker=kwargs.get("marker", "o"),
            s=kwargs.get("markersize", 5),
            label=kwargs.get("label"),
            zorder=kwargs.get("zorder", TwoLevelMesh.ZORDERS["vertices"]),
        )

    @staticmethod
    def plot_edges(
        ax: Axes,
        edges: list,  # list of edges
        mesh,
        **kwargs,
    ):
        """
        Plot a single mesh edge on a matplotlib axis.

        Args:
            ax (Axes): Matplotlib axis to plot on.
            edge: EdgeId of the edge to plot.
            mesh: NGSolve mesh containing the edge.
            color (str, optional): Color for the edge. Defaults to "black".
            linewidth (float, optional): Width of the edge line. Defaults to 1.5.
            linestyle (str, optional): Style of the edge line. Defaults to "-".
            label (str, optional): Label for the edge (optional).
        """
        for edge in edges:
            vertices = [mesh[v].point for v in mesh[edge].vertices]
            ax.plot(
                [v[0] for v in vertices],
                [v[1] for v in vertices],
                color=kwargs.get("color", "black"),
                linewidth=kwargs.get("linewidth", 1.5),
                linestyle=kwargs.get("linestyle", "-"),
                label=kwargs.get("label"),
                zorder=kwargs.get("zorder", TwoLevelMesh.ZORDERS["edges"]),
            )

    @staticmethod
    def plot_faces(
        ax: Axes,
        faces: list,  # list of faces
        mesh,
        **kwargs,
    ):
        for face in faces:
            # NOTE: this is only for 3D meshes
            raise NotImplementedError(
                "Plotting faces is not implemented yet. This is only used for 3D meshes."
            )

    def visualize_two_level_mesh(self, show: bool = True) -> Figure:
        """
        Visualize the two-level mesh with subdomains and layers.

        Args:
            ax (Axes): Matplotlib axis to plot on.

        Returns:
            Figure: The matplotlib figure with the plotted two-level mesh.
        """
        LOGGER.info("Visualizing TwoLevelMesh")
        from hcmsfem.plot_utils import set_mpl_style

        set_mpl_style()
        fig, ax = plt.subplots(3, 2, figsize=(8, 12), sharex=True, sharey=True)
        LOGGER.debug("\tcreated figure and axes...")

        self.plot_mesh(ax[0, 0], mesh_type="fine")
        self.plot_mesh(ax[0, 0], mesh_type="coarse")
        ax[0, 0].set_title("Fine and Coarse Meshes")
        LOGGER.debug("\tplotted fine and coarse meshes.")

        # plot the domains
        self.plot_domains(ax[0, 1], domains=3, plot_layers=True)
        ax[0, 1].set_title("Subdomains and Layers")
        LOGGER.debug("\tplotted subdomains and layers.")

        # plot all connected components
        self.plot_connected_components(ax[1, 0])
        ax[1, 0].set_title("Interface")
        LOGGER.debug("\tplotted connected components.")

        # plot the connected component tree
        self.plot_connected_component_tree(ax[1, 1])
        ax[1, 1].set_title("Connected Component Tree")
        LOGGER.debug("\tplotted connected component tree.")

        # plot slab elements (single)
        self.plot_mesh(ax[2, 0], mesh_type="coarse")
        self.plot_edge_slabs(ax[2, 0], which="double")
        ax[2, 0].set_title("Edge Slabs (Double)")
        LOGGER.debug("\tplotted edge slabs (double).")

        # plot slab elements (double)
        self.plot_mesh(ax[2, 1], mesh_type="coarse")
        self.plot_edge_slabs(ax[2, 1], which="vertices")
        ax[2, 1].set_title("Edge Slabs (Around Vertices)")
        LOGGER.debug("\tplotted edge slabs (around vertices).")

        fig.tight_layout()
        LOGGER.debug("\tvisualization complete.")

        if show:
            LOGGER.debug("\tshowing figure...")
            plt.show()

        return fig

    @property
    def subdomain_colors(self) -> cycle:
        """
        List of colors for plotting domains.
        """
        if not hasattr(self, "_subdomain_colors"):
            self._subdomain_colors = cycle(CUSTOM_COLORS_SIMPLE)
        return self._subdomain_colors

    @subdomain_colors.setter
    def subdomain_colors(self, colors):
        """
        Set custom colors for the domains.

        Args:
            colors (list): List of color strings.
        """
        if not isinstance(colors, list):
            raise ValueError("Domain colors must be a list.")
        self._subdomain_colors = cycle(colors)


##################
# usage examples #
##################
class TwoLevelMeshExamples:

    MESH_PARAMS = MeshParams(lx=1.0, ly=1.0, Nc=4, refinement_levels=4, layers=2)
    SAVE_DIR = TwoLevelMesh.get_save_dir(MESH_PARAMS)

    @classmethod
    def example_creation(cls, fig_toggle: bool = True):
        two_mesh = TwoLevelMesh(mesh_params=cls.MESH_PARAMS)
        two_mesh.save()  # Save the mesh and subdomains
        if fig_toggle:
            _ = two_mesh.visualize_two_level_mesh(show=True)

    @classmethod
    def example_load(cls):
        _ = TwoLevelMesh.load(cls.MESH_PARAMS)

    # Profiling the mesh creation & loading
    @classmethod
    def profile(cls, creation: bool = True, loading: bool = True, top: int = 10):
        import cProfile
        import pstats
        from hcmsfem.visualize_profile import visualize_profile
        if creation:
            fp = cls.SAVE_DIR / "mesh_creation.prof"
            cProfile.run(
                "TwoLevelMeshExamples.example_creation(fig_toggle=False)", str(fp)
            )
            p_creation = pstats.Stats(str(fp))
            p_creation.sort_stats("cumulative").print_stats(top)
            visualize_profile(fp)

        if loading:
            fp = cls.SAVE_DIR / "mesh_loading.prof"
            cProfile.run("TwoLevelMeshExamples.example_load()", str(fp))
            p_loading = pstats.Stats(str(fp))
            p_loading.sort_stats("cumulative").print_stats(top)
            visualize_profile(fp)


if __name__ == "__main__":
    TwoLevelMeshExamples.example_creation(
        fig_toggle=True
    )  # Uncomment to create and (optionally) visualize a new mesh
    TwoLevelMeshExamples.example_load()  # Uncomment to load an existing mesh
    # TwoLevelMeshExamples.profile()  # Uncomment to profile the mesh creation & loading
