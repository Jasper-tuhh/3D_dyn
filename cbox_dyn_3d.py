"""
cbox_dyn_3d.py -- DYNAMIC version of the 3D cbox.

The prescribed corner displacement of the quasi-static model is replaced by a time-dependent
FORCE; the structure is released at t = T_OFF (force = 0) and runs freely until T_END.

    M a + f_int(u) = f_ext(t),     u(0) = v(0) = a(0) = 0

Time integration: Newmark-beta (default beta = 1/4, gamma = 1/2: average acceleration,
second order, unconditionally stable, no numerical damping).  The balance is evaluated at t_{n+1}
and solved for u_{n+1} with Newton:
    rho * a_{n+1} . w  +  dPi(u_{n+1}; w)  +  regularisation(u_{n+1}; w)  -  f_ext(t_{n+1}) . w  = 0
    a_{n+1} = (u_{n+1} - u_n - dt v_n - dt^2 (1/2 - beta) a_n) / (beta dt^2)
    v_{n+1} = v_n + dt ((1 - gamma) a_n + gamma a_{n+1})
Numerical damping (optional): gamma > 1/2 with beta = (gamma + 1/2)^2 / 4, e.g. NEWMARK_GAMMA=0.6
(this reduces the accuracy to first order).
Failed Newton steps are retried with dt/2 (up to MAX_SPLIT times).

Load: vertical traction  t = (0, -LOAD_AMP * s(t), 0)  on the top face strip
      x in [1 - LOAD_WIDTH, 1], y = 0.5   (the tip of the upper arm), all z.
      s(t) is zero at t = 0 and again at t = T_OFF:
        FORCE_PROFILE = halfsine (default) : sin(pi t / T_OFF)
                        sin2               : sin^2(pi t / T_OFF)
                        triangle           : linear up to T_OFF/2, linear down to 0
Material/geometry identical to cbox_3d.py (E = 1, nu = 0.4 in the 2D units, RHO = 1 -> the first
bending period of the upper arm is ~60 s, i.e. the 25 s / 50 s window fits).

Environment (all optional)
  T_END=50 T_OFF=25 DT=0.25 FORCE_PROFILE=halfsine LOAD_AMP=3e-3 LOAD_WIDTH=0.1
  RHO=1 RHO_M=0 (density of the third medium; 0 = massless, behaves like the void it replaces)
  NEWMARK_GAMMA=0.5 NEWMARK_BETA=(gamma+0.5)^2/4 MAX_SPLIT=4 WRITE_EVERY=4 MATERIAL_MODEL ZBC NZ DEPTH CELL_TYPE
  REG_METHODS REG_VALUE P_VALUES MESHES NEWTON_MAXIT/ATOL/RTOL/STOL
"""
import os

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import default_scalar_type, geometry
from dolfinx.io import VTXWriter
from dolfinx.mesh import locate_entities, locate_entities_boundary, meshtags
from dolfinx.fem import (Constant, Function, dirichletbc, functionspace,
                         locate_dofs_topological, assemble_scalar, form)
from dolfinx.fem.petsc import create_matrix, create_vector
from ufl import (Measure, TestFunction, derivative, inner, as_vector)

from methods import make_material, huhu_lulu_regularization, huhu_regularization
from functional import emit as _emit, print_hardware_info, NonlinearPDE_SNESProblem
from functional_3d import kinematics_3d, psi_3d, psi_2_3d
from driver3d import make_box_mesh, z_support_bcs, env_list, parse_meshes

# ---- configuration -----------------------------------------------------------------------
MATERIAL_MODEL = os.environ.get("MATERIAL_MODEL", "neo_hooke")
CELL_TYPE = os.environ.get("CELL_TYPE", "hexahedron")
ZBC = os.environ.get("ZBC", "free")
NZ = int(os.environ.get("NZ", "8"))
L_LAYER = 0.1
DEPTH = float(os.environ["DEPTH"]) if "DEPTH" in os.environ else 1.0 + L_LAYER

T_END = float(os.environ.get("T_END", "50"))
T_OFF = float(os.environ.get("T_OFF", "25"))
DT = float(os.environ.get("DT", "0.25"))
FORCE_PROFILE = os.environ.get("FORCE_PROFILE", "halfsine")
LOAD_AMP = float(os.environ.get("LOAD_AMP", "3e-3"))      # traction amplitude (force / area)
LOAD_WIDTH = float(os.environ.get("LOAD_WIDTH", "0.1"))   # strip length in x at the arm tip
RHO = float(os.environ.get("RHO", "1.0"))
RHO_M = float(os.environ.get("RHO_M", "0.0"))
GAM_NM = float(os.environ.get("NEWMARK_GAMMA", "0.5"))
BETA_NM = float(os.environ.get("NEWMARK_BETA", str(0.25 * (GAM_NM + 0.5) ** 2)))
MAX_SPLIT = int(os.environ.get("MAX_SPLIT", "4"))
WRITE_EVERY = int(os.environ.get("WRITE_EVERY", "4"))

MU, K, GAMMA = 5 / 14, 5 / 3, 1.0e-7
TAG_LOAD = 5

def force_factor(t):
    """s(t): 0 at t=0, 0 for t >= T_OFF."""
    if t <= 0.0 or t >= T_OFF:
        return 0.0
    x = t / T_OFF
    if FORCE_PROFILE == "halfsine":
        return float(np.sin(np.pi * x))
    if FORCE_PROFILE == "sin2":
        return float(np.sin(np.pi * x) ** 2)
    if FORCE_PROFILE == "triangle":
        return float(1.0 - abs(2.0 * x - 1.0))
    raise ValueError(f"unknown FORCE_PROFILE '{FORCE_PROFILE}'")


def make_probe(mesh, point):
    """Evaluate a vector Function at a fixed point (collective)."""
    tdim = mesh.topology.dim
    tree = geometry.bb_tree(mesh, tdim)
    p = np.zeros((1, 3))
    p[0, :len(point)] = point
    cand = geometry.compute_collisions_points(tree, p)
    cells = geometry.compute_colliding_cells(mesh, cand, p).links(0)
    n_local = mesh.topology.index_map(tdim).size_local
    cell = np.array([c for c in cells if c < n_local][:1], dtype=np.int32)

    def probe(f):
        val = f.eval(p, cell).ravel().copy() if len(cell) > 0 else None
        vals = [v for v in mesh.comm.allgather(val) if v is not None]
        return vals[0] if vals else np.full(3, np.nan)
    return probe


def run_case(Nx, Ny, fe_degree, reg_method, k_reg, out_dir, emit):
    comm = MPI.COMM_WORLD
    t_wall = MPI.Wtime()
    if fe_degree < 2:
        raise ValueError("Hessian regularisation needs p >= 2")
    if Nx % 10 or Ny % 5:
        raise ValueError("cbox needs Nx % 10 == 0 and Ny % 5 == 0")
    force_factor(0.5 * T_OFF)          # validates FORCE_PROFILE early

    # ---- mesh, tags (same as cbox_3d.py) ----
    Nx_tot = Nx + int(round(Nx * L_LAYER))
    Lx, Ly = 1.0 + L_LAYER, 0.5
    mesh0 = make_box_mesh(comm, Lx, Ly, DEPTH, Nx_tot, Ny, NZ, CELL_TYPE)
    tdim = mesh0.topology.dim
    fdim = tdim - 1
    mesh0.topology.create_connectivity(fdim, tdim)
    imap = mesh0.topology.index_map(tdim)
    num_local_cells = imap.size_local
    tol = 1e-8
    marker = np.zeros(imap.size_local + imap.num_ghosts, dtype=np.int32)
    marker[locate_entities(mesh0, tdim, lambda x: (x[0] >= 0.1 - tol) & (x[1] >= 0.1 - tol)
                           & (x[1] <= 0.4 + tol))] = 2
    body = np.concatenate([
        locate_entities(mesh0, tdim, lambda x: x[0] <= 0.1 + tol),
        locate_entities(mesh0, tdim, lambda x: (x[1] <= 0.1 + tol) & (x[0] <= 1.0 + tol)),
        locate_entities(mesh0, tdim, lambda x: (x[1] >= 0.4 - tol) & (x[0] <= 1.0 + tol))])
    for c in body:
        if marker[c] == 0:
            marker[c] = 1
    marker[marker == 0] = 2
    material_tags = meshtags(mesh0, tdim, np.arange(num_local_cells, dtype=np.int32),
                             marker[:num_local_cells])

    # ---- load strip on the top face at the tip of the upper arm ----
    load_facets = locate_entities_boundary(
        mesh0, fdim, lambda x: np.isclose(x[1], Ly) & (x[0] >= 1.0 - LOAD_WIDTH - tol)
        & (x[0] <= 1.0 + tol))
    load_facets = np.sort(load_facets).astype(np.int32)
    facet_tags = meshtags(mesh0, fdim, load_facets,
                          np.full(len(load_facets), TAG_LOAD, dtype=np.int32))

    metadata = {"quadrature_degree": 2 * fe_degree + 2}
    dx = Measure("dx", domain=mesh0, subdomain_data=material_tags, metadata=metadata)
    ds = Measure("ds", domain=mesh0, subdomain_data=facet_tags, metadata=metadata)

    # ---- spaces and state ----
    Vu = functionspace(mesh0, ("Lagrange", fe_degree, (mesh0.geometry.dim,)))
    w = TestFunction(Vu)
    u = Function(Vu, name="displacement")        # unknown u_{n+1}
    u_n = Function(Vu)
    v_n = Function(Vu, name="velocity")
    a_n = Function(Vu)

    # ---- boundary conditions: fixed left wall (+ optional plane strain) ----
    left_facets = locate_entities_boundary(mesh0, fdim, lambda x: np.isclose(x[0], 0.0))
    bcl = dirichletbc(np.zeros(3, dtype=default_scalar_type),
                      locate_dofs_topological(Vu, fdim, left_facets), Vu)
    bcs = [bcl] + z_support_bcs(mesh0, Vu, DEPTH, ZBC)

    # ---- Newmark residual at t_{n+1} ----
    dt_c = Constant(mesh0, default_scalar_type(DT))
    fac_c = Constant(mesh0, default_scalar_type(0.0))
    mat = make_material(MU, K, MATERIAL_MODEL)
    k = default_scalar_type(k_reg)

    kin = kinematics_3d(u)
    Pi = psi_3d(mat, kin) * dx(1) + GAMMA * psi_2_3d(mat, kin) * dx(2)
    R_reg = (huhu_lulu_regularization(u, w, dx, k, k, 2) if reg_method == "huhu_lulu"
             else huhu_regularization(u, w, dx, k, 2))
    R_int = derivative(Pi, u, w) + R_reg

    a_new = (u - u_n - dt_c * v_n - dt_c ** 2 * (0.5 - BETA_NM) * a_n) / (BETA_NM * dt_c ** 2)
    R_in = RHO * inner(a_new, w) * dx(1)
    if RHO_M > 0.0:
        R_in = R_in + RHO_M * inner(a_new, w) * dx(2)
    t_ext = as_vector((0.0, -LOAD_AMP * fac_c, 0.0))
    R = R_in + R_int - inner(t_ext, w) * ds(TAG_LOAD)

    # ---- diagnostics forms ----
    Pi_u = Pi                      # strain energy of the current state (same expression)
    KE = 0.5 * RHO * inner(v_n, v_n) * dx(1)
    if RHO_M > 0.0:
        KE = KE + 0.5 * RHO_M * inner(v_n, v_n) * dx(2)
    SE_form, KE_form = form(Pi_u), form(KE)
    load_area = comm.allreduce(assemble_scalar(form(
        Constant(mesh0, default_scalar_type(1.0)) * ds(TAG_LOAD))), op=MPI.SUM)
    probe = make_probe(mesh0, (1.0, Ly, 0.5 * DEPTH))     # tip of the upper arm

    dof_count = Vu.dofmap.index_map.size_global * Vu.dofmap.index_map_bs
    num_cells = imap.size_global
    peak_force = LOAD_AMP * load_area
    emit(f"--- cbox DYNAMIC p={fe_degree}, {Nx_tot}x{Ny}x{NZ} {CELL_TYPE}, {num_cells} cells, "
         f"{dof_count} dofs, {reg_method} k={k_reg:.1e}, {MATERIAL_MODEL}, ZBC={ZBC} ---")
    emit(f"    T_END={T_END:g}, T_OFF={T_OFF:g}, DT={DT:g}, profile={FORCE_PROFILE}, "
         f"rho={RHO:g}, rho_medium={RHO_M:g}, Newmark beta={BETA_NM:.4f}, gamma={GAM_NM:.3f}")
    if GAM_NM < 0.5 or BETA_NM < 0.25 * (GAM_NM + 0.5) ** 2 - 1e-12:
        emit("    WARNING: beta < (gamma+1/2)^2/4 or gamma < 1/2 -> only conditionally stable")
    emit(f"    peak traction {LOAD_AMP:g} on strip area {load_area:.4g} -> peak force {peak_force:.4g} "
         f"({peak_force / DEPTH:.4g} per unit depth)")

    # ---- Newton solver (assembled once, reused for every step) ----
    problem = NonlinearPDE_SNESProblem(R, u, bcs)
    b = create_vector(Vu)
    J_L = create_matrix(problem.a)
    snes = PETSc.SNES().create(comm)
    snes.setOptionsPrefix("dyn_")
    snes.setFunction(problem.F, b)
    snes.setJacobian(problem.J, J_L)
    snes.setTolerances(atol=float(os.environ.get("NEWTON_ATOL", "1e-12")),
                       rtol=float(os.environ.get("NEWTON_RTOL", "1e-8")),
                       stol=float(os.environ.get("NEWTON_STOL", "1e-10")),
                       max_it=int(os.environ.get("NEWTON_MAXIT", "15")))
    snes.getKSP().setType("preonly")
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")
    opts = PETSc.Options()
    opts["dyn_snes_linesearch_type"] = "bt"
    opts["dyn_snes_max_linear_solve_fail"] = 20
    opts["dyn_snes_linesearch_max_it"] = 20
    snes.setFromOptions()

    hist = []                       # (t, s(t), tip u_y, KE, SE, newton its)
    state = {"t": 0.0, "steps": 0, "newton": 0, "failed": 0, "min_dt": DT}

    def record(its):
        ke = comm.allreduce(assemble_scalar(KE_form), op=MPI.SUM)
        se = comm.allreduce(assemble_scalar(SE_form), op=MPI.SUM)
        tip = probe(u_n)
        hist.append((state["t"], force_factor(state["t"]), float(tip[1]), ke, se, its))

    def attempt(t, dt):
        dt_c.value = dt
        fac_c.value = force_factor(t + dt)
        # Newmark predictor (a_{n+1} = 0) as starting guess; satisfies the homogeneous BCs
        u.x.array[:] = u_n.x.array + dt * v_n.x.array + dt ** 2 * (0.5 - BETA_NM) * a_n.x.array
        u.x.scatter_forward()
        x = u.x.petsc_vec.copy()
        try:
            x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            snes.solve(None, x)
            reason = snes.getConvergedReason()
            its = snes.getIterationNumber()
            if reason <= 0:
                emit(f"  t={t:.4f} dt={dt:.4g}: Newton failed (reason={reason}, its={its}, "
                     f"|F|={snes.getFunctionNorm():.2e})")
                return False, its
            x.copy(u.x.petsc_vec)
            u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        except Exception as e:
            emit(f"  t={t:.4f} dt={dt:.4g}: solver exception: {e}")
            return False, 0
        finally:
            x.destroy()
        # Newmark update of acceleration and velocity (uses the OLD u_n, v_n, a_n)
        a_new_v = (u.x.array - u_n.x.array - dt * v_n.x.array
                   - dt ** 2 * (0.5 - BETA_NM) * a_n.x.array) / (BETA_NM * dt ** 2)
        v_new_v = v_n.x.array + dt * ((1 - GAM_NM) * a_n.x.array + GAM_NM * a_new_v)
        a_n.x.array[:] = a_new_v
        v_n.x.array[:] = v_new_v
        u_n.x.array[:] = u.x.array
        for f in (a_n, v_n, u_n):
            f.x.scatter_forward()
        return True, its

    def advance(t, dt, level):
        ok, its = attempt(t, dt)
        if ok:
            state["t"] = t + dt
            state["steps"] += 1
            state["newton"] += its
            state["min_dt"] = min(state["min_dt"], dt)
            record(its)
            if state["steps"] % WRITE_EVERY == 0:
                writer.write(state["t"])
            if state["steps"] % 10 == 0:
                h_ = hist[-1]
                emit(f"  t={h_[0]:7.3f}  s={h_[1]:.3f}  tip u_y={h_[2]:+.5f}  KE={h_[3]:.3e}  "
                     f"SE={h_[4]:.3e}  its={its}")
            return True
        u.x.array[:] = u_n.x.array
        u.x.scatter_forward()
        state["failed"] += 1
        if level >= MAX_SPLIT:
            emit(f"  t={t:.4f}: dt={dt:.3e} still failing after {MAX_SPLIT} splits -> abort")
            return False
        half = 0.5 * dt
        return advance(t, half, level + 1) and advance(t + half, half, level + 1)

    time_setup = MPI.Wtime() - t_wall
    writer = VTXWriter(comm, os.path.join(out_dir, "results.bp"), [u_n, v_n], engine="BP4")
    completed = True
    try:
        record(0)
        writer.write(0.0)
        while state["t"] < T_END - 1e-9:
            dt = min(DT, T_END - state["t"])
            if not advance(state["t"], dt, 0):
                completed = False
                break
        writer.write(state["t"])
    finally:
        writer.close()
        snes.destroy()
        J_L.destroy()
        b.destroy()

    h = np.array(hist)
    if comm.rank == 0:
        np.savetxt(os.path.join(out_dir, "history.csv"), h, delimiter=",",
                   header="t,force_factor,tip_uy,kinetic_energy,strain_energy,newton_its", comments="")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
            ax[0].plot(h[:, 0], h[:, 2], "b-", label="tip $u_y$")
            ax[0].set_ylabel("tip displacement $u_y$")
            a2 = ax[0].twinx()
            a2.plot(h[:, 0], h[:, 1] * peak_force, "r--", label="force")
            a2.set_ylabel("applied force")
            ax[0].axvline(T_OFF, color="k", lw=0.5)
            ax[1].plot(h[:, 0], h[:, 3], label="kinetic")
            ax[1].plot(h[:, 0], h[:, 4], label="strain (hyperelastic + medium)")
            ax[1].plot(h[:, 0], h[:, 3] + h[:, 4], "k:", label="sum")
            ax[1].axvline(T_OFF, color="k", lw=0.5)
            ax[1].set_xlabel("t"); ax[1].set_ylabel("energy per model"); ax[1].legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "history.png"), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[history plot skipped: {e}]")

    return {"completed": completed, "t_final": state["t"], "steps": state["steps"],
            "newton": state["newton"], "failed": state["failed"], "min_dt": state["min_dt"],
            "dof": int(dof_count), "num_cells": int(num_cells),
            "max_tip": float(np.nanmin(h[:, 2])),
            "time_total": comm.allreduce(MPI.Wtime() - t_wall, op=MPI.MAX),
            "time_setup": comm.allreduce(time_setup, op=MPI.MAX)}


def main(comm=MPI.COMM_WORLD):
    reg_methods = env_list("REG_METHODS", "huhu_lulu")
    degrees = env_list("P_VALUES", "2", int)
    meshes = parse_meshes(os.environ.get("MESHES", "20x10"))
    k_reg = float(os.environ.get("REG_VALUE", "1e-5"))
    if comm.rank == 0:
        print_hardware_info()
    for reg_method in reg_methods:
        assert reg_method in ("huhu_lulu", "huhu"), reg_method
        name = f"cbox_dyn3d_{reg_method}"
        dir_name = os.path.join(".", name)
        if comm.rank == 0:
            os.makedirs(dir_name, exist_ok=True)
        comm.Barrier()
        log_path = os.path.join(dir_name, "terminal.txt")

        def emit(msg, _lp=log_path):
            _emit(_lp, msg, comm)

        _emit(log_path, f"{name}: k={k_reg:.1e}, degrees={degrees}, meshes={meshes}", comm, mode="w")
        for fe_degree in degrees:
            for (Nx, Ny) in meshes:
                out_dir = os.path.join(dir_name, f"p{fe_degree}_{Nx}x{Ny}")
                if comm.rank == 0:
                    os.makedirs(out_dir, exist_ok=True)
                comm.Barrier()
                res = run_case(Nx, Ny, fe_degree, reg_method, k_reg, out_dir, emit)
                emit(f"=> {'completed' if res['completed'] else 'ABORTED'} at t={res['t_final']:.3f}: "
                     f"{res['steps']} steps, {res['newton']} Newton its "
                     f"({res['newton'] / max(res['steps'], 1):.2f}/step), {res['failed']} failed attempts, "
                     f"min dt {res['min_dt']:.3g}, extreme tip u_y = {res['max_tip']:.4f}, "
                     f"{res['time_total']:.1f} s")


if __name__ == "__main__":
    main()