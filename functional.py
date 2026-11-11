"""FE tooling: elements, kinematics, SNES wrapper, stress post-processing, utilities."""
import os
import platform
import time
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import basix
import basix.ufl
from ufl import (Identity, variable, as_tensor, grad, tr, det, TrialFunction,
                 derivative, sqrt)
from dolfinx import geometry, default_real_type
from dolfinx.fem import Function, Expression, functionspace, form
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix, apply_lifting, set_bc)


# ============================================================================
# Utilities
# ============================================================================
def emit(log_path, line="", comm=MPI.COMM_WORLD, mode="a", echo=True):
    """Print (optional) and append `line` to `log_path` on rank 0 only."""
    if comm.rank == 0:
        if echo:
            print(line, flush=True)
        with open(log_path, mode) as f:
            f.write(str(line) + "\n")


def print_hardware_info():
    try:
        import psutil
    except ImportError:
        psutil = None
    print("\n--- Hardware information ---")
    print(f"Local Time      : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"OS              : {platform.system()} {platform.release()}")
    print(f"Architecture    : {platform.machine()}")
    print(f"Processor       : {platform.processor()}")
    print(f"CPU cores       : {os.cpu_count()}")
    if psutil is not None:
        print(f"Physical cores  : {psutil.cpu_count(logical=False)}")
        print(f"Total memory    : {psutil.virtual_memory().total / 1024**3:.2f} GB")
    print(f"Python          : {platform.python_version()}")
    print("-----------------------------\n")


def von_mises(sigma):
    s11, s22, s33 = sigma[0, 0], sigma[1, 1], sigma[2, 2]
    s12, s13, s23 = sigma[0, 1], sigma[0, 2], sigma[1, 2]
    return sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2
                       + 6 * (s12 ** 2 + s13 ** 2 + s23 ** 2)))


def get_interpolation_points(elem):
    ip = elem.interpolation_points
    return ip() if callable(ip) else ip


def find_dof_by_coordinates(V, target_coords):
    X = V.tabulate_dof_coordinates()
    return np.argmin((X[:, 0] - target_coords[0]) ** 2 + (X[:, 1] - target_coords[1]) ** 2)


def find_node_by_coordinates(mesh, target_coords):
    c = mesh.geometry.x
    return np.argmin(np.sqrt((c[:, 0] - target_coords[0]) ** 2 + (c[:, 1] - target_coords[1]) ** 2))


def cell_mean(f):
    """Cell-wise mean of a DG function (one value per cell)."""
    return f.x.array[f.function_space.dofmap.list].mean(axis=1)


# ============================================================================
# Elements
# ============================================================================
def make_element(cell, degree, family="Lagrange", variant="gll_warped",
                 dpc_variant="diagonal_gll", shape=None, discontinuous=False):
    """Basix element on a quadrilateral cell.
    family  : "Lagrange" (Q_p) or "serendipity" (S_p)
    variant : Lagrange variant = node distribution, e.g. "gll_warped" (GLL points),
              "equispaced", "chebyshev_warped", ... (names of basix.LagrangeVariant)
    discontinuous=True gives the DG version of the same element."""
    kw = dict(shape=shape, discontinuous=discontinuous, dtype=default_real_type,
              lagrange_variant=basix.LagrangeVariant[variant])
    fam = family.lower()
    if fam in ("lagrange", "q", "p"):
        name = "Lagrange"
    else:
        raise ValueError(f"Unknown element family '{family}'")
    return basix.ufl.element(name, cell, degree, **kw)

# ============================================================================
# Kinematics (plane strain embedded in 3x3); all UFL variables for diff(psi, F)
# ============================================================================
def kinematics(u):
    dim = len(u)
    I = variable(Identity(dim))
    F_0 = variable(I + grad(u))
    F = variable(as_tensor([[F_0[0, 0], F_0[0, 1], 0],
                            [F_0[1, 0], F_0[1, 1], 0],
                            [0, 0, 1]]))
    C = variable(F.T * F)
    Ic = variable(tr(C))
    J = variable(det(F))
    Cb = variable(J ** (-2.0 / 3.0) * C)
    Ib1 = variable(J ** (-2.0 / 3.0) * Ic)
    Ib2 = variable(0.5 * (Ib1 ** 2 - tr(Cb * Cb)))
    return SimpleNamespace(F=F, J=J, Ic=Ic, Ib1=Ib1, Ib2=Ib2)


def cauchy_stress(psi_expr, kin):
    from ufl import diff
    return (1.0 / kin.J) * diff(psi_expr, kin.F) * kin.F.T


# ============================================================================
# SNES wrapper
# ============================================================================
class NonlinearPDE_SNESProblem:
    def __init__(self, F, u, bcs):
        V = u.function_space
        du = TrialFunction(V)
        self.L = form(F)
        self.a = form(derivative(F, u, du))
        self.bcs = bcs
        self.u = u

    def F(self, snes, x, F):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                       mode=PETSc.ScatterMode.FORWARD)
        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self.L)
        apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], alpha=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, self.bcs, x, -1.0)

    def J(self, snes, x, J, P):
        J.zeroEntries()
        assemble_matrix(J, self.a, bcs=self.bcs)
        J.assemble()


# ============================================================================
# Stress post-processing (discontinuous fields of any degree + point evaluation)
# ============================================================================
class StressPostProcessor:
    """sigma22 and von Mises of each region interpolated into a DG space, plus
    'total' fields that pick the right region per cell. DG is used because the
    target node sits on the half-space / third-medium interface."""

    REGION_OF_TAG = {1: "el", 2: "cylinder", 3: "medium"}

    def __init__(self, mesh, cell_tags, sigma_el, sigma_cyl, sigma_med, element):
        self.mesh = mesh
        Vs = functionspace(mesh, element)
        self.Vs = Vs
        pts = get_interpolation_points(Vs.element)

        sig = {"el": sigma_el, "cylinder": sigma_cyl, "medium": sigma_med}
        self.exprs = {k: Expression(e[1, 1], pts) for k, e in sig.items()}
        self.vm_exprs = {k: Expression(von_mises(e), pts) for k, e in sig.items()}

        self.f = {k: Function(Vs, name=f"sigma22_{k}") for k in sig}
        self.f["total"] = Function(Vs, name="sigma22_total")
        self.fvm = {k: Function(Vs, name=f"vm_{k}") for k in sig}
        self.vm_total = Function(Vs, name="von_mises")
        self.fields = [self.f["el"], self.f["cylinder"], self.f["medium"],
                       self.f["total"], self.vm_total]

        tdim = mesh.topology.dim
        imap = mesh.topology.index_map(tdim)
        self.tag_of_cell = np.zeros(imap.size_local + imap.num_ghosts, dtype=np.int32)
        self.tag_of_cell[cell_tags.indices] = cell_tags.values

        n = Vs.dofmap.index_map.size_local + Vs.dofmap.index_map.num_ghosts
        self.tag_of_dof = np.zeros(n, dtype=np.int32)
        self.tag_of_dof[Vs.dofmap.list[cell_tags.indices]] = cell_tags.values[:, None]
        self.tree = geometry.bb_tree(mesh, tdim)

    def _combine(self, parts, total):
        arr = total.x.array
        arr[:] = 0.0
        for tag, name in self.REGION_OF_TAG.items():
            m = self.tag_of_dof == tag
            arr[m] = parts[name].x.array[m]

    def update(self):
        for k in self.exprs:
            self.f[k].interpolate(self.exprs[k])
            self.fvm[k].interpolate(self.vm_exprs[k])
        self._combine(self.f, self.f["total"])
        self._combine(self.fvm, self.vm_total)
        for f in self.fields:
            f.x.scatter_forward()

    def value_at(self, point, prefer_tag=1, field="total", comm=MPI.COMM_WORLD):
        """Value of a stress field at a point; on interfaces, cells with
        `prefer_tag` win. Returns None if the point is not found."""
        p = np.zeros((1, 3))
        p[0, :len(point)] = point
        cand = geometry.compute_collisions_points(self.tree, p)
        cells = geometry.compute_colliding_cells(self.mesh, cand, p).links(0)
        n_local = self.mesh.topology.index_map(self.mesh.topology.dim).size_local
        cells = [c for c in cells if c < n_local]
        val, cnt = 0.0, 0
        if cells:
            pref = [c for c in cells if self.tag_of_cell[c] == prefer_tag]
            c = (pref or cells)[0]
            val = float(self.f[field].eval(p, np.array([c], dtype=np.int32)).ravel()[0])
            cnt = 1
        val = comm.allreduce(val, op=MPI.SUM)
        cnt = comm.allreduce(cnt, op=MPI.SUM)
        return val / cnt if cnt else None