"""
monolith_dyn_3d.py -- DYNAMIC version of the 3D monolith (hollow frame with a third-medium cavity).

The prescribed top-midpoint displacement of monolith_3d.py is replaced by a time-dependent FORCE
on a strip at the top-centre; the force vanishes at t = T_OFF and the frame then runs freely to
T_END.  The run is NOT stopped at the first contact of the cavity walls: the closing gap is
monitored, time steps are refined while the gap is small, and the simulation continues through
and beyond the contact phase.

    M a + f_int(u) = f_ext(t),     u(0) = v(0) = a(0) = 0

Newmark-beta (default beta = 1/4, gamma = 1/2: average acceleration, 2nd order, unconditionally
stable, no numerical damping; NEWMARK_GAMMA=0.6 adds damping, beta = (gamma+1/2)^2/4):
    rho a_{n+1} . w + dPi(u_{n+1}; w) + regularisation(u_{n+1}; w) - f_ext(t_{n+1}) . w = 0
    a_{n+1} = (u_{n+1} - u_n - dt v_n - dt^2 (1/2 - beta) a_n) / (beta dt^2)
    v_{n+1} = v_n + dt ((1 - gamma) a_n + gamma a_{n+1})
A failed Newton step is retried with dt/2 (up to MAX_SPLIT times).

Contact monitor (cavity height 0.3 at rest):
    gap(t) = 0.3 + u_y(1.0, 0.4, z_mid) - u_y(1.0, 0.1, z_mid)     (inner faces at mid-span)
    "first contact" = first time gap <= CONTACT_TOL.  For gap < GAP_BAND the step is DT_CONTACT.
    T_AFTER_CONTACT > 0 stops the run that many seconds after first contact (otherwise T_END).

Load: vertical traction (0, -LOAD_AMP s(t), 0) on the top face strip |x - 1| <= LOAD_WIDTH, y = 0.5.
      s(t) = 0 at t = 0 and t >= T_OFF:  halfsine (default) | sin2 | triangle
Supports as in monolith_3d.py: left edge (0,0,z) fixed, right edge (2,0,z) u_y = 0.

Environment (optional)
  T_END=25 T_OFF=10 DT=0.05 DT_CONTACT=0.025 FORCE_PROFILE=halfsine LOAD_AMP=2.0 LOAD_WIDTH=0.1
  CONTACT_TOL=0.03 GAP_BAND=0.1 T_AFTER_CONTACT=0 RHO=1 RHO_M=0 NEWMARK_GAMMA=0.5 NEWMARK_BETA
  MAX_SPLIT=4 WRITE_EVERY=4 MATERIAL_MODEL(hartmann_neff) ZBC NZ DEPTH CELL_TYPE
  REG_METHODS REG_VALUE P_VALUES MESHES  NEWTON_MAXIT/ATOL/RTOL/STOL
Rough scales for RHO=1 (estimates): global bending period ~2.4 s, top flange ~6 s.
"""
import os

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import default_scalar_type, geometry
from dolfinx.io import VTXWriter
from dolfinx.mesh import locate_entities, locate_entities_boundary, meshtags
from dolfinx.fem import (Constant, Function, dirichletbc, functionspace,
                         assemble_scalar, form)
from dolfinx.fem.petsc import create_matrix, create_vector
from ufl import Measure, TestFunction, derivative, inner, as_vector

from methods import make_material, huhu_lulu_regularization, huhu_regularization
from functional import emit as _emit, print_hardware_info, NonlinearPDE_SNESProblem
from functional_3d import kinematics_3d, psi_3d, psi_2_3d
from driver3d import make_box_mesh, dofs_on_edges, z_support_bcs, env_list, parse_meshes

# ---- configuration -----------------------------------------------------------------------
MATERIAL_MODEL = os.environ.get("MATERIAL_MODEL", "hartmann_neff")
CELL_TYPE = os.environ.get("CELL_TYPE", "hexahedron")
ZBC = os.environ.get("ZBC", "free")
NZ = int(os.environ.get("NZ", "8"))
LX, LY = 2.0, 0.5
DEPTH = float(os.environ["DEPTH"]) if "DEPTH" in os.environ else LX
GAP0 = 0.3                                      # cavity height at rest (y from 0.1 to 0.4)

T_END = float(os.environ.get("T_END", "25"))
T_OFF = float(os.environ.get("T_OFF", "10"))
DT = float(os.environ.get("DT", "0.05"))
DT_CONTACT = float(os.environ.get("DT_CONTACT", str(0.5 * DT)))
FORCE_PROFILE = os.environ.get("FORCE_PROFILE", "halfsine")
LOAD_AMP = float(os.environ.get("LOAD_AMP", "2.0"))        # traction amplitude (force / area)
LOAD_WIDTH = float(os.environ.get("LOAD_WIDTH", "0.1"))    # half-width of the strip around x = 1
CONTACT_TOL = float(os.environ.get("CONTACT_TOL", "0.03"))
GAP_BAND = float(os.environ.get("GAP_BAND", "0.1"))
T_AFTER_CONTACT = float(os.environ.get("T_AFTER_CONTACT", "0"))
RHO = float(os.environ.get("RHO", "1.0"))
RHO_M = float(os.environ.get("RHO_M", "0.0"))
GAM_NM = float(os.environ.get("NEWMARK_GAMMA", "0.5"))
BETA_NM = float(os.environ.get("NEWMARK_BETA", str(0.25 * (GAM_NM + 0.5) ** 2)))
MAX_SPLIT = int(os.environ.get("MAX_SPLIT", "4"))
WRITE_EVERY = int(os.environ.get("WRITE_EVERY", "4"))

MU, K, GAMMA = 10.0, 20.0, 1.0e-7
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
    if Nx % 20 or Ny % 5:
        raise ValueError("monolith needs Nx % 20 == 0 and Ny % 5 == 0")
    force_factor(0.5 * T_OFF)          # validates FORCE_PROFILE early

    # ---- mesh and tags (same rules as monolith_3d.py): 1 = frame, 3 = third medium ----
    mesh0 = make_box_mesh(comm, LX, LY, DEPTH, Nx, Ny, NZ, CELL_TYPE)
    tdim = mesh0.topology.dim
    fdim = tdim - 1
    mesh0.topology.create_connectivity(fdim, tdim)
    imap = mesh0.topology.index_map(tdim)
    num_local_cells = imap.size_local
    tol = 1e-8
    marker = np.zeros(imap.size_local + imap.num_ghosts, dtype=np.int32)
    marker[locate_entities(mesh0, tdim, lambda x: (x[0] >= 0.1 - tol) & (x[0] <= LX - 0.1 + tol)
                           & (x[1] >= 0.1 - tol) & (x[1] <= 0.4 + tol))] = 3
    body = locate_entities(mesh0, tdim, lambda x: (x[0] <= 0.1 + tol) | (x[0] >= LX - 0.1 - tol)
                           | (x[1] <= 0.1 + tol) | (x[1] >= 0.4 - tol))
    for c in body:
        if marker[c] == 0:
            marker[c] = 1
    material_tags = meshtags(mesh0, tdim, np.arange(num_local_cells, dtype=np.int32),
                             marker[:num_local_cells])

    # ---- load strip at the top centre ----
    load_facets = locate_entities_boundary(
        mesh0, fdim, lambda x: np.isclose(x[1], LY) & (np.abs(x[0] - 0.5 * LX) <= LOAD_WIDTH + tol))
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

    # ---- boundary conditions: supports only (no prescribed displacement) ----
    left_edge = lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    right_edge = lambda x: np.isclose(x[0], LX) & np.isclose(x[1], 0.0)
    bcl = dirichletbc(np.zeros(3, dtype=default_scalar_type),
                      dofs_on_edges(mesh0, Vu, left_edge), Vu)
    zero = Constant(mesh0, default_scalar_type(0.0))
    bcr = dirichletbc(zero, dofs_on_edges(mesh0, Vu.sub(1), right_edge), Vu.sub(1))
    bcs = [bcl, bcr] + z_support_bcs(mesh0, Vu, DEPTH, ZBC)

    # ---- Newmark residual at t_{n+1} ----
    dt_c = Constant(mesh0, default_scalar_type(DT))
    fac_c = Constant(mesh0, default_scalar_type(0.0))
    mat = make_material(MU, K, MATERIAL_MODEL)
    k = default_scalar_type(k_reg)

    kin = kinematics_3d(u)
    Pi = psi_3d(mat, kin) * dx(1) + GAMMA * psi_2_3d(mat, kin) * dx(3)
    R_reg = (huhu_lulu_regularization(u, w, dx, k, k, 3) if reg_method == "huhu_lulu"
             else huhu_regularization(u, w, dx, k, 3))
    R_int = derivative(Pi, u, w) + R_reg

    a_new = (u - u_n - dt_c * v_n - dt_c ** 2 * (0.5 - BETA_NM) * a_n) / (BETA_NM * dt_c ** 2)
    R_in = RHO * inner(a_new, w) * dx(1)
    if RHO_M > 0.0:
        R_in = R_in + RHO_M * inner(a_new, w) * dx(3)
    t_ext = as_vector((0.0, -LOAD_AMP * fac_c, 0.0))
    R = R_in + R_int - inner(t_ext, w) * ds(TAG_LOAD)

    # ---- diagnostics ----
    KE = 0.5 * RHO * inner(v_n, v_n) * dx(1)
    if RHO_M > 0.0:
        KE = KE + 0.5 * RHO_M * inner(v_n, v_n) * dx(3)
    SE_form, KE_form = form(Pi), form(KE)
    load_area = comm.allreduce(assemble_scalar(form(
        Constant(mesh0, default_scalar_type(1.0)) * ds(TAG_LOAD))), op=MPI.SUM)
    zmid = 0.5 * DEPTH
    probe_top_in = make_probe(mesh0, (0.5 * LX, 0.4, zmid))      # underside of the top wall
    probe_bot_in = make_probe(mesh0, (0.5 * LX, 0.1, zmid))      # inner face of the bottom wall
    probe_load = make_probe(mesh0, (0.5 * LX, LY, zmid))         # loaded point

    dof_count = Vu.dofmap.index_map.size_global * Vu.dofmap.index_map_bs
    num_cells = imap.size_global
    peak_force = LOAD_AMP * load_area
    emit(f"--- monolith DYNAMIC p={fe_degree}, {Nx}x{Ny}x{NZ} {CELL_TYPE}, {num_cells} cells, "
         f"{dof_count} dofs, {reg_method} k={k_reg:.1e}, {MATERIAL_MODEL}, ZBC={ZBC} ---")
    emit(f"    T_END={T_END:g}, T_OFF={T_OFF:g}, DT={DT:g} (DT_CONTACT={DT_CONTACT:g} for gap<{GAP_BAND:g}), "
         f"profile={FORCE_PROFILE}, rho={RHO:g}, rho_medium={RHO_M:g}, "
         f"Newmark beta={BETA_NM:.4f}, gamma={GAM_NM:.3f}")
    emit(f"    peak traction {LOAD_AMP:g} on strip area {load_area:.4g} -> peak force {peak_force:.4g} "
         f"({peak_force / DEPTH:.4g} per unit depth); contact when gap <= {CONTACT_TOL:g}")
    if GAM_NM < 0.5 or BETA_NM < 0.25 * (GAM_NM + 0.5) ** 2 - 1e-12:
        emit("    WARNING: beta < (gamma+1/2)^2/4 or gamma < 1/2 -> only conditionally stable")

    # ---- Newton solver (assembled once, reused for every step) ----
    problem = NonlinearPDE_SNESProblem(R, u, bcs)
    b = create_vector(Vu)
    J_L = create_matrix(problem.a)
    snes = PETSc.SNES().create(comm)
    snes.setOptionsPrefix("dyn_")
    snes.setFunction(problem.F, b)
    snes.setJacobian(problem.J, J_L)
    snes.setTolerances(atol=float(os.environ.get("NEWTON_ATOL", "1e-8")),
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

    hist = []        # t, s(t), load-point u_y, gap, KE, SE, newton its
    state = {"t": 0.0, "t_end": T_END, "steps": 0, "newton": 0, "failed": 0, "min_dt": DT,
             "gap": GAP0, "t_contact": None, "min_gap": GAP0}

    def record(its):
        ke = comm.allreduce(assemble_scalar(KE_form), op=MPI.SUM)
        se = comm.allreduce(assemble_scalar(SE_form), op=MPI.SUM)
        gap = GAP0 + float(probe_top_in(u_n)[1]) - float(probe_bot_in(u_n)[1])
        state["gap"] = gap
        state["min_gap"] = min(state["min_gap"], gap)
        if state["t_contact"] is None and gap <= CONTACT_TOL:
            state["t_contact"] = state["t"]
            if T_AFTER_CONTACT > 0.0:
                state["t_end"] = min(T_END, state["t"] + T_AFTER_CONTACT)
            emit(f"  *** first contact at t={state['t']:.4f} (gap={gap:.4f}); "
                 f"simulation continues until t={state['t_end']:g}")
        hist.append((state["t"], force_factor(state["t"]), float(probe_load(u_n)[1]),
                     gap, ke, se, its))

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
                emit(f"  t={h_[0]:7.3f}  s={h_[1]:.3f}  load pt u_y={h_[2]:+.5f}  gap={h_[3]:.4f}  "
                     f"KE={h_[4]:.3e}  SE={h_[5]:.3e}  its={its}")
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
        while state["t"] < state["t_end"] - 1e-9:
            dt = DT_CONTACT if state["gap"] < GAP_BAND else DT
            dt = min(dt, state["t_end"] - state["t"])
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
        np.savetxt(os.path.join(out_dir, "history.csv"), h, delimiter=",", comments="",
                   header="t,force_factor,load_point_uy,gap,kinetic_energy,strain_energy,newton_its")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
            ax[0].plot(h[:, 0], h[:, 1] * peak_force, "r-")
            ax[0].set_ylabel("applied force")
            ax[1].plot(h[:, 0], h[:, 2], "b-", label="load point $u_y$")
            ax[1].plot(h[:, 0], h[:, 3], "g-", label="cavity gap")
            ax[1].axhline(CONTACT_TOL, color="g", ls=":", lw=0.8)
            ax[1].legend(); ax[1].set_ylabel("displacement / gap")
            ax[2].plot(h[:, 0], h[:, 4], label="kinetic")
            ax[2].plot(h[:, 0], h[:, 5], label="strain (hyperelastic + medium)")
            ax[2].plot(h[:, 0], h[:, 4] + h[:, 5], "k:", label="sum")
            ax[2].set_xlabel("t"); ax[2].set_ylabel("energy per model"); ax[2].legend()
            for a_ in ax:
                a_.axvline(T_OFF, color="k", lw=0.5)
                if state["t_contact"] is not None:
                    a_.axvline(state["t_contact"], color="m", lw=0.8, ls="--")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "history.png"), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[history plot skipped: {e}]")

    return {"completed": completed, "t_final": state["t"], "steps": state["steps"],
            "newton": state["newton"], "failed": state["failed"], "min_dt": state["min_dt"],
            "t_contact": state["t_contact"], "min_gap": state["min_gap"],
            "dof": int(dof_count), "num_cells": int(num_cells),
            "extreme_uy": float(np.nanmin(h[:, 2])),
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
        name = f"monolith_dyn3d_{reg_method}"
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
                tc = res["t_contact"]
                emit(f"=> {'completed' if res['completed'] else 'ABORTED'} at t={res['t_final']:.3f}: "
                     f"{res['steps']} steps, {res['newton']} Newton its "
                     f"({res['newton'] / max(res['steps'], 1):.2f}/step), {res['failed']} failed attempts, "
                     f"min dt {res['min_dt']:.3g}")
                emit(f"   first contact: {'none (gap never <= %g)' % CONTACT_TOL if tc is None else 't=%.3f' % tc}, "
                     f"min gap {res['min_gap']:.4f}, extreme load-point u_y {res['extreme_uy']:.4f}, "
                     f"{res['time_total']:.1f} s")


if __name__ == "__main__":
    main()