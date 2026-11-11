"""
mesh3d.py -- structured hexahedral mesh: a TRUE sphere on a substrate, void filled
with a third medium.  Quarter model (x>=0, y>=0), pure hexahedra, no gmsh.

Topology ("cubed sphere" in one tensor-product grid)
----------------------------------------------------
The domain above the substrate is the box [0,L]x[0,L]x[0,z_top] containing a sphere
of radius R centred at (0,0,zc), zc = R + gap0.  Grid coordinates q in [0,1]x[0,1]x[-1,1]:

    m = max(|q|_inf)          m <= a : core  -> ball (tag 2)
                              m  > a : shell -> third medium (tag 3)
    core   (m<=a): cube blended into the sphere  (no singular axis, no degenerate cells)
    shell  (m> a): linear blend  sphere point  ->  box-wall point
                   sphere point = c + R * n(q/m),  n = equi-angular cube->sphere map
                   wall point   = (L qx/m, L qy/m, zmap(qz/m))   (conforming to the box)
    below z=0     : substrate (tag 1), same plan grid as the wall z=0.

Mesh size: `h` is the element size ON THE SPHERE SURFACE AT THE CONTACT POLE; it grows
geometrically (grow_core) towards the equator / north pole, so the sphere surface is
finest exactly where contact happens.  Substrate plan grid and sphere grid share the
same grid lines (fine under the pole, coarse far away).

Facet tags: 10 bottom (u_z=0)  12 plane x=0  13 plane y=0   (exterior facets)
            11 loaded cap on the sphere's north pole (INTERIOR facets between ball and
               third medium: load it with dS and u('+'); u is continuous so '+'/'-' agree)
"""
from dataclasses import dataclass
import numpy as np

TAG_SUBSTRATE, TAG_BALL, TAG_MEDIUM = 1, 2, 3
TAG_BOTTOM, TAG_LOAD, TAG_SYM_X, TAG_SYM_Y = 10, 11, 12, 13


@dataclass
class Geometry3D:
    R: float = 1.0          # sphere radius
    gap0: float = 0.05      # initial gap at the south pole (2D: 0.05)
    L: float = 1.5          # lateral half-size of the domain
    H_sub: float = 2.0      # substrate depth (2D: 2.0)
    z_top: float = 2.3      # domain top; sphere top is at gap0 + 2R = 2.05
    w_load: float = 0.25    # half-width of the loaded cap patch around the north pole
    n_shell: int = 6        # element layers of third medium between sphere and walls
    grow_core: float = 1.12     # growth of surface element size away from the contact pole
    grow_sub: float = 1.25      # growth of substrate layers with depth
    beta: float = 0.7           # half-size of the inner cube of the ball, in units of R
    a_core: float = None        # core extent in q (default: matches plan grids at the pole)

    def __post_init__(self):
        if self.a_core is None:
            self.a_core = float(np.clip(np.pi * self.R / (4 * self.L), 0.35, 0.85))
        if self.L <= self.R:
            raise ValueError("L must exceed R")
        if self.z_top <= self.gap0 + 2 * self.R:
            raise ValueError("z_top must lie above the top of the sphere")
        if not 0 < self.w_load < self.R:
            raise ValueError("need 0 < w_load < R")

    @property
    def zc(self):
        return self.R + self.gap0


def _geom_nodes(a, b, n, ratio=1.0):
    """n+1 nodes from a to b, element sizes grow by `ratio` from a towards b."""
    w = ratio ** np.arange(n)
    t = np.concatenate(([0.0], np.cumsum(w))) / w.sum()
    return a + (b - a) * t


def _n_graded(length, h0, ratio):
    """Number of elements so that the first element is <= h0 with growth `ratio`."""
    if ratio <= 1.0:
        return max(1, int(np.ceil(length / h0)))
    return max(1, int(np.ceil(np.log(1 + length * (ratio - 1) / h0) / np.log(ratio))))


class Layout:
    """Grid tables and the map  unit cube -> physical coordinates."""

    def __init__(self, geo, h):
        g = geo
        self.geo = g
        a, nsh, r = g.a_core, g.n_shell, g.grow_core
        dq1 = h / (g.R * np.pi / 4)          # first surface cell in q^ units (pole: x ~ R*pi/4*q^)
        self.n_cx = _n_graded(1.0, dq1, r)   # core cells in x and y  (q^ in [0,1])
        self.n_cz = _n_graded(2.0, dq1, r)   # core cells in z        (q^ in [-1,1])
        self.qh_x = _geom_nodes(0.0, 1.0, self.n_cx, r)
        qh_z = -1.0 + 2.0 * _geom_nodes(0.0, 1.0, self.n_cz, r)
        self.qx = np.concatenate((a * self.qh_x, np.linspace(a, 1, nsh + 1)[1:]))
        self.qz = np.concatenate((np.linspace(-1, -a, nsh + 1)[:-1], a * qh_z,
                                  np.linspace(a, 1, nsh + 1)[1:]))
        self.Ni = self.Nj = self.n_cx + nsh
        self.n_sub = max(3, _n_graded(g.H_sub, h, g.grow_sub))
        self.Nc = nsh + self.n_cz + nsh
        self.Nk = self.n_sub + self.Nc
        self.t_sub = 1.0 - _geom_nodes(0, 1, self.n_sub, g.grow_sub)[::-1]
        self.k_core0 = self.n_sub + nsh                 # first core layer (south)
        self.k_core1 = self.n_sub + nsh + self.n_cz     # end of core layers (north pole plane)
        # loaded cap: largest index whose surface point (on the x-axis) has x <= w_load
        xs = g.R * np.sin(np.pi / 4 * self.qh_x)
        self.n_w = max(1, int(np.sum(xs <= g.w_load + 1e-12)) - 1)
        self.w_real = float(xs[self.n_w])

    @property
    def shape(self):
        return self.Ni, self.Nj, self.Nk

    def _idx(self, xi):
        return (np.rint(xi[:, 0] * self.Ni).astype(int),
                np.rint(xi[:, 1] * self.Nj).astype(int),
                np.rint(xi[:, 2] * self.Nk).astype(int))

    def _cube_to_phys(self, q):
        g = self.geo
        a, c = g.a_core, np.array([0.0, 0.0, g.zc])
        m = np.abs(q).max(axis=1)
        mm = np.where(m > 0, m, 1.0)
        qh = q / mm[:, None]                       # point on the unit-cube surface
        t_ = np.tan(np.pi / 4 * qh)
        n = t_ / np.linalg.norm(t_, axis=1)[:, None]          # equi-angular cube -> sphere
        Rn = g.R * n
        zmap = np.where(qh[:, 2] >= 0, g.zc + qh[:, 2] * (g.z_top - g.zc), g.zc * (1 + qh[:, 2]))
        B = np.column_stack((g.L * qh[:, 0], g.L * qh[:, 1], zmap))   # box wall point
        X = np.empty_like(q)
        core = m <= a
        t = (m[core] / a)[:, None]
        X[core] = c + t * ((1 - t) * g.beta * g.R * qh[core] + t * Rn[core])
        sh = ~core
        s = ((m[sh] - a) / (1 - a))[:, None]
        X[sh] = (1 - s) * (c + Rn[sh]) + s * B[sh]
        return X

    def map_nodes(self, xi):
        g = self.geo
        i, j, k = self._idx(xi)
        X = np.empty((len(i), 3))
        sub = k < self.n_sub
        X[sub, 0] = g.L * self.qx[i[sub]]
        X[sub, 1] = g.L * self.qx[j[sub]]
        X[sub, 2] = -g.H_sub + self.t_sub[k[sub]] * g.H_sub
        up = ~sub
        q = np.column_stack((self.qx[i[up]], self.qx[j[up]], self.qz[k[up] - self.n_sub]))
        X[up] = self._cube_to_phys(q)
        return X

    def cell_tags(self, xi_mid):
        ic = np.floor(xi_mid[:, 0] * self.Ni).astype(int)
        jc = np.floor(xi_mid[:, 1] * self.Nj).astype(int)
        kc = np.floor(xi_mid[:, 2] * self.Nk).astype(int)
        tags = np.full(len(ic), TAG_MEDIUM, dtype=np.int32)
        tags[kc < self.n_sub] = TAG_SUBSTRATE
        ball = ((ic < self.n_cx) & (jc < self.n_cx)
                & (kc >= self.k_core0) & (kc < self.k_core1))
        tags[ball] = TAG_BALL
        return tags


@dataclass
class MeshData3D:
    mesh: object
    cell_tags: object
    facet_tags: object
    layout: Layout


def generate_mesh_3d(mesh_h, comm, geo=None):
    """Same role as generate_mesh() in the 2D code: returns .mesh/.cell_tags/.facet_tags."""
    from dolfinx.mesh import (create_box, CellType, meshtags,
                              locate_entities, locate_entities_boundary)

    geo = geo or Geometry3D()
    lay = Layout(geo, mesh_h)
    mesh = create_box(comm, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], list(lay.shape),
                      CellType.hexahedron)
    tdim, fdim = mesh.topology.dim, mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, tdim)

    # everything below is evaluated on the *unit cube*, before the nodes are moved
    ncell = mesh.topology.index_map(tdim).size_local
    xg = mesh.geometry.x
    mid = xg[mesh.geometry.dofmap[:ncell]].mean(axis=1)
    cell_tags = meshtags(mesh, tdim, np.arange(ncell, dtype=np.int32), lay.cell_tags(mid))

    eps = 1e-9
    xs = lay.n_w / lay.Ni + eps
    ys = lay.n_w / lay.Nj + eps
    zload = lay.k_core1 / lay.Nk
    idx, val = [], []
    for tag, mk, loc in (
        (TAG_BOTTOM, lambda x: np.isclose(x[2], 0.0), locate_entities_boundary),
        (TAG_SYM_X, lambda x: np.isclose(x[0], 0.0), locate_entities_boundary),
        (TAG_SYM_Y, lambda x: np.isclose(x[1], 0.0), locate_entities_boundary),
        (TAG_LOAD, lambda x: np.isclose(x[2], zload, atol=1e-9) & (x[0] <= xs) & (x[1] <= ys),
         locate_entities),                      # interior facets: ball / third medium
    ):
        f = loc(mesh, fdim, mk)
        idx.append(f)
        val.append(np.full(len(f), tag, dtype=np.int32))
    idx, val = np.concatenate(idx), np.concatenate(val)
    order = np.argsort(idx)
    facet_tags = meshtags(mesh, fdim, idx[order].astype(np.int32), val[order])

    mesh.geometry.x[:] = lay.map_nodes(mesh.geometry.x)
    return MeshData3D(mesh, cell_tags, facet_tags, lay)