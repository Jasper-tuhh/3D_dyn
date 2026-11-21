"""
hertz_3d_dyn.py -- 3D dynamic ball bounce on an elastic substrate (third-medium contact),
Newmark time integration.  Extension of the quasi-static 3D Hertz script.

Library files: mesh3d.py, methods.py, functional.py, functional_3d.py, plotting.py
(stress3d.py / driver3d.py are not needed; functional_3d.StressPostProcessor3D is used.)

What is new w.r.t. the static script
  * void between ball and substrate is one sphere radius (gap0 = R)
  * ball is released from rest and accelerated by gravity g = 9.81 (body force on the ball)
  * Newmark-beta/gamma (default gamma = 0.6, beta = (gamma+1/2)^2/4 -> mild numerical damping)
        a_{n+1} = (u - u_n - dt v_n) / (beta dt^2) - (1/(2 beta) - 1) a_n
        v_{n+1} = v_n + dt ((1-gamma) a_n + gamma a_{n+1})
    a_{n+1} is a UFL expression of the unknown u, so the SNES Jacobian contains the
    mass term automatically (derivative of the residual).
  * consistent initial acceleration a_0 from  M a_0 = f_ext  (u_0 = v_0 = 0)
  * adaptive time step: coarse in free flight, fine near/in contact, halved on Newton failure
  * Newmark predictor as Newton initial guess
  * output: results_u.bp (u, v), results_stress.bp, trajectory.csv, trajectory.png

Environment variables (all optional)
  H_VALUES 0.05        P_VALUES 2            MATERIAL_MODEL neo_hooke
  REG_METHOD huhu_lulu REG_VALUE 1e-3
  T_END 1.5            DT_FREE 4e-3          DT_CONTACT 5e-4     DT_MIN 1e-6
  GAP_FINE 0.25        (dt = DT_CONTACT when the gap below the pole is < GAP_FINE)
  RHO_BALL 1.0         RHO_SUB 1.0           RHO_MEDIUM 1e-3
  G 9.81               NEWMARK_GAMMA 0.6     ALPHA_M 0.0 (mass-proportional damping)
  N_SHELL 6            WRITE_EVERY 5         VERBOSE 0
Units are consistent but arbitrary (as in the static script: E_ball ~ 7e3, E_sub ~ 7e4);
choose RHO_BALL so that the impact is neither rigid nor trivially soft.
"""
import os
import traceback
from contextlib import ExitStack

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from ufl import (Measure, TestFunction, TrialFunction, SpatialCoordinate,
                 derivative, inner, dot, grad)
from dolfinx import default_scalar_type, geometry
from dolfinx.io import VTXWriter
from dolfinx.fem import (Constant, Function, dirichletbc, functionspace, form,
                         locate_dofs_topological, assemble_scalar)
from dolfinx.fem.petsc import (create_matrix, create_vector, assemble_matrix,
                               assemble_vector, apply_lifting, set_bc)

from methods import (make_material, young_poisson,
                     huhu_lulu_regularization, huhu_regularization)
from functional import (emit as _emit, print_hardware_info, make_element,
                        NonlinearPDE_SNESProblem)
from functional_3d import (kinematics_3d, psi_3d, psi_2_3d, cauchy_stress_3d,
                           StressPostProcessor3D)
from plotting import create_plots
from mesh3d import (Geometry3D, generate_mesh_3d,
                    TAG_BOTTOM, TAG_SYM_X, TAG_SYM_Y)

comm = MPI.COMM_WORLD


def env_f(name, default):
    return float(os.environ.get(name, default))


h_values = [float(x) for x in os.environ.get("H_VALUES", "0.05").split(",")]
p_values = [int(x) for x in os.environ.get("P_VALUES", "2").split(",")]

# ---- Material --------------------------------------------------------------
MATERIAL_MODEL = os.environ.get("MATERIAL_MODEL", "neo_hooke")
mu_c = default_scalar_type(35000 / 13); K_c = default_scalar_type(17500 / 3)     # ball
mu_e = default_scalar_type(700000 / 29); K_e = default_scalar_type(700000 / 3)   # substrate
mat_c = make_material(mu_c, K_c, MATERIAL_MODEL)
mat_e = make_material(mu_e, K_e, MATERIAL_MODEL)
gamma_m = default_scalar_type(1.3e-3)          # third-medium scaling (was `gamma` in static script)

# ---- Dynamics ----------------------------------------------------------------
G = env_f("G", 9.81)
RHO_BALL = env_f("RHO_BALL", 1.0)
RHO_SUB = env_f("RHO_SUB", 1.0)
RHO_MEDIUM = env_f("RHO_MEDIUM", 1e-3)          # third medium: (almost) massless
NM_GAMMA = env_f("NEWMARK_GAMMA", 0.6)          # 0.5 = no numerical damping
NM_BETA = 0.25 * (NM_GAMMA + 0.5) ** 2
ALPHA_M = env_f("ALPHA_M", 0.0)                 # mass-proportional (Rayleigh) damping
T_END = env_f("T_END", 1.5)
DT_FREE = env_f("DT_FREE", 4e-3)
DT_CONTACT = env_f("DT_CONTACT", 5e-4)
DT_MIN = env_f("DT_MIN", 1e-6)
GAP_FINE = env_f("GAP_FINE", 0.25)
WRITE_EVERY = int(os.environ.get("WRITE_EVERY", "5"))
VERBOSE = os.environ.get("VERBOSE", "0") == "1"

# ---- Geometry: void = one radius between sphere and substrate ----------------------
R_SPHERE = 1.0
GAP0 = R_SPHERE                                  # <- "one radius between ball and surface"
GEO = Geometry3D(R=R_SPHERE, gap0=GAP0, z_top=GAP0 + 2 * R_SPHERE + 0.3,
                 n_shell=int(os.environ.get("N_SHELL", "6")))
target_node = (0.0, 0.0, 0.0)                    # contact centre on the substrate surface

# ---- Regularisation / solver -------------------------------------------------------
reg_method = os.environ.get("REG_METHOD", "huhu_lulu")
assert reg_method in ("huhu_lulu", "huhu"), reg_method
k_reg = env_f("REG_VALUE", "1e-3")
ELEMENT = dict(family="Lagrange", variant="gll_warped")
STRESS_ELEMENT = dict(degree=0, variant="gll_warped")
QUADRATURE = dict(rule="GLL", degree="auto")     # GLL => diagonal (lumped) mass matrix
linesearch = "bt"
max_it_newton = 15
max_attempts = 12

# =============================================================================
sweep_name = f"hertz3d_dyn_{reg_method}_{MATERIAL_MODEL}"
sweep_dir = os.path.join(".", sweep_name)
case_name = f"hertz3d_dyn_k{k_reg:.1e}_{reg_method}"
sweep_log_path = os.path.join(sweep_dir, "sweep_terminal.txt")
_case_log = None

if comm.rank == 0:
    os.makedirs(sweep_dir, exist_ok=True)
    with open(sweep_log_path, "w") as f:
        f.write("3D Dynamic Hertzian Bounce Log\n" + "=" * 40 + "\n")
comm.Barrier()


def emit(msg):
    if _case_log is not None:
        _emit(_case_log, msg, comm)
    _emit(sweep_log_path, msg, comm, echo=_case_log is None)


def hertz_impact(m_full, v_imp, R, mat_ball, mat_sub):
    """Small-strain Hertz energy balance 1/2 m v^2 = 2/5 K d^(5/2), K = 4/3 E* sqrt(R)."""
    E1, n1 = young_poisson(mat_ball)
    E2, n2 = young_poisson(mat_sub)
    Es = 1.0 / ((1 - n1 ** 2) / E1 + (1 - n2 ** 2) / E2)
    K = 4.0 / 3.0 * Es * np.sqrt(R)
    d = (5.0 * m_full * v_imp ** 2 / (4.0 * K)) ** 0.4
    F = K * d ** 1.5
    a = np.sqrt(R * d)
    p0 = 3 * F / (2 * np.pi * a ** 2)
    return dict(delta=d, F=F, a=a, p0=p0)


def make_point_eval(mesh):
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    n_local = mesh.topology.index_map(mesh.topology.dim).size_local

    def ev(f, point):
        p = np.zeros((1, 3))
        p[0, :len(point)] = point
        cand = geometry.compute_collisions_points(tree, p)
        cells = [c for c in geometry.compute_colliding_cells(mesh, cand, p).links(0)
                 if c < n_local]
        val = f.eval(p, np.array(cells[:1], dtype=np.int32)).ravel().copy() if cells else None
        vals = [x for x in comm.allgather(val) if x is not None]
        return vals[0] if vals else np.full(3, np.nan)
    return ev


def run_case(mesh_h, fe_degree, k_huhu, k_lulu, write_outputs=True, comm=MPI.COMM_WORLD):
    global _case_log
    t_start = MPI.Wtime()

    case_dir = os.path.join(sweep_dir, f"{case_name}_p{fe_degree}_h{mesh_h:g}")
    if comm.rank == 0:
        os.makedirs(case_dir, exist_ok=True)
        with open(os.path.join(case_dir, "terminal.txt"), "w") as f:
            f.write(f"3D dynamic Hertz bounce: p={fe_degree}, h={mesh_h:g}, "
                    f"material={MATERIAL_MODEL}\n")
    comm.Barrier()
    _case_log = os.path.join(case_dir, "terminal.txt")

    if fe_degree < 2:
        raise ValueError("Hessian-based regularisation needs p >= 2.")

    # ---- Mesh ----
    mesh_data = generate_mesh_3d(mesh_h=mesh_h, comm=comm, geo=GEO)
    mesh0, material_tags, facet_tags = mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags
    tdim = mesh0.topology.dim
    fdim = tdim - 1
    mesh0.topology.create_connectivity(fdim, tdim)
    mesh0.topology.create_connectivity(0, tdim)

    q_deg = 2 * fe_degree - 1 if QUADRATURE["degree"] == "auto" else int(QUADRATURE["degree"])
    metadata = {"quadrature_rule": "GLL", "quadrature_degree": q_deg}
    dx = Measure("dx", domain=mesh0, subdomain_data=material_tags, metadata=metadata)

    # ---- Space and state ----
    cell = mesh0.basix_cell()
    P_u = make_element(cell, fe_degree, shape=(mesh0.geometry.dim,), **ELEMENT)
    V = functionspace(mesh0, P_u)
    w = TestFunction(V)
    u = Function(V, name="u")          # unknown u_{n+1}
    u_n = Function(V)                  # converged state
    v_n = Function(V, name="v")
    a_n = Function(V)

    # ---- BCs (quarter model, roller support) ----
    zero = Constant(mesh0, default_scalar_type(0))
    bcs_def = [(2, TAG_BOTTOM), (0, TAG_SYM_X), (1, TAG_SYM_Y)]
    bcu = [dirichletbc(zero, locate_dofs_topological(V.sub(c), fdim, facet_tags.find(tag)), V.sub(c))
           for c, tag in bcs_def]

    # ---- Newmark constants ----
    dt_c = Constant(mesh0, default_scalar_type(DT_FREE))
    beta, gam = NM_BETA, NM_GAMMA
    c_a = 1.0 / (2.0 * beta) - 1.0
    a_new = (u - u_n - dt_c * v_n) / (beta * dt_c ** 2) - c_a * a_n
    v_new = v_n + dt_c * ((1.0 - gam) * a_n + gam * a_new)
    g_vec = Constant(mesh0, np.array([0.0, 0.0, -G], dtype=default_scalar_type))

    def mass(a, wt):
        return (RHO_SUB * inner(a, wt) * dx(1) + RHO_BALL * inner(a, wt) * dx(2)
                + RHO_MEDIUM * inner(a, wt) * dx(3))

    # ---- Energy / residual ----
    kin = kinematics_3d(u)
    psi_e = psi_3d(mat_e, kin)
    psi_c = psi_3d(mat_c, kin)
    psi_m = gamma_m * psi_2_3d(mat_c, kin)
    Pi_el = psi_e * dx(1) + psi_c * dx(2) + psi_m * dx(3)

    if reg_method == "huhu_lulu":
        R_reg = huhu_lulu_regularization(u, w, dx, k_huhu, k_lulu, 3)
    else:
        R_reg = huhu_regularization(u, w, dx, k_huhu, 3)

    inertia = mass(a_new, w)
    if ALPHA_M > 0.0:
        inertia = inertia + ALPHA_M * mass(v_new, w)
    gravity = RHO_BALL * inner(g_vec, w) * dx(2)          # self-weight of the ball only
    L = derivative(Pi_el, u, w) + R_reg + inertia - gravity

    # ---- Diagnostics forms ----
    X = SpatialCoordinate(mesh0)
    f_mass = form(RHO_BALL * dx(2))
    f_z = form(RHO_BALL * (X[2] + u_n[2]) * dx(2))
    f_vz = form(RHO_BALL * v_n[2] * dx(2))
    f_az = form(RHO_BALL * a_n[2] * dx(2))
    f_kin = form(0.5 * (RHO_SUB * dot(v_n, v_n) * dx(1) + RHO_BALL * dot(v_n, v_n) * dx(2)
                        + RHO_MEDIUM * dot(v_n, v_n) * dx(3)))
    f_el = form(Pi_el)       # evaluated with u (= u_n after an accepted step)

    def asm(f):
        return comm.allreduce(assemble_scalar(f), op=MPI.SUM)

    m_q = asm(f_mass)                     # quarter-model ball mass
    m_full = 4.0 * m_q

    # ---- Stress output ----
    stress_el = make_element(cell, fe_degree, family="Lagrange",
                             variant=STRESS_ELEMENT["variant"], discontinuous=True)
    post = StressPostProcessor3D(mesh0, material_tags,
                                 cauchy_stress_3d(psi_e, kin), cauchy_stress_3d(psi_c, kin),
                                 cauchy_stress_3d(psi_m, kin), stress_el)
    ev = make_point_eval(mesh0)
    south_pole = (0.0, 0.0, GAP0)

    def gap_now():
        return GAP0 + ev(u_n, south_pole)[2] - ev(u_n, target_node)[2]

    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    num_cells = mesh0.topology.index_map(tdim).size_global
    v_imp = np.sqrt(2.0 * G * GAP0)
    ref = hertz_impact(m_full, v_imp, GEO.R, mat_c, mat_e)
    emit(f"--- p={fe_degree}, h={mesh_h:g}: {num_cells} hex cells, {num_dofs} dofs, quadrature {metadata} ---")
    emit(f"    Newmark beta={beta:.4f} gamma={gam:.3f}, ball mass (full) = {m_full:.4f}, "
         f"free-fall impact speed ~ {v_imp:.3f}, fall time ~ {np.sqrt(2 * GAP0 / G):.3f}")
    emit(f"    Hertz energy-balance reference: delta_max={ref['delta']:.4f}, F_max={ref['F']:.3f}, "
         f"a={ref['a']:.4f}, p0={ref['p0']:.2f}")

    # ---- Consistent initial acceleration: M a0 = f_ext (u0 = v0 = 0) ----
    a_tr = TrialFunction(V)
    a_form = form(mass(a_tr, w))
    A0 = create_matrix(a_form)
    assemble_matrix(A0, a_form, bcs=bcu)
    A0.assemble()
    b0 = create_vector(V)
    with b0.localForm() as bl:
        bl.set(0.0)
    f_form = form(RHO_BALL * inner(g_vec, w) * dx(2))
    assemble_vector(b0, f_form)
    apply_lifting(b0, [a_form], bcs=[bcu])
    b0.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b0, bcu)
    ksp0 = PETSc.KSP().create(comm)
    ksp0.setOperators(A0)
    ksp0.setType("preonly")
    ksp0.getPC().setType("lu")
    ksp0.getPC().setFactorSolverType("mumps")
    ksp0.solve(b0, a_n.x.petsc_vec)
    a_n.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    ksp0.destroy(); A0.destroy(); b0.destroy()
    a0_com = asm(f_az) / m_q
    emit(f"    initial ball acceleration a_z = {a0_com:.4f} (expected {-G:.4f})")

    # ---- SNES ----
    problem = NonlinearPDE_SNESProblem(L, u, bcs=bcu)
    b = create_vector(V)
    J_L = create_matrix(problem.a)
    snes = PETSc.SNES().create(comm)
    snes.setOptionsPrefix("hertzdyn_")
    snes.setFunction(problem.F, b)
    snes.setJacobian(problem.J, J_L)
    snes.setTolerances(atol=1e-8, rtol=1e-9, stol=1e-10, max_it=max_it_newton)
    snes.getKSP().setType("preonly")
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")
    opts = PETSc.Options()
    opts["hertzdyn_snes_linesearch_type"] = linesearch
    opts["hertzdyn_snes_max_linear_solve_fail"] = 20
    opts["hertzdyn_snes_linesearch_max_it"] = 20
    if VERBOSE:
        opts["hertzdyn_snes_monitor"] = None
        opts["hertzdyn_snes_converged_reason"] = None
    snes.setFromOptions()

    # ---- Time loop ----
    stats = {"newton": 0, "steps": 0, "failed": 0}
    hist = []                      # rows: t, z_com, v_com, gap, Fc, sigma_zz, E_kin, E_el, E_grav
    t, n, dt = 0.0, 0, DT_FREE
    gap = GAP0
    min_hits, aborted = 0, False
    stack = ExitStack()
    time_snes = 0.0
    csv_path = os.path.join(case_dir, "trajectory.csv")
    if comm.rank == 0:
        with open(csv_path, "w") as f:
            f.write("t,z_com,v_com,gap,F_contact_full,sigma_zz_target,E_kin,E_el,E_grav,E_total\n")
    try:
        if write_outputs:
            vtx_u = stack.enter_context(VTXWriter(
                mesh0.comm, os.path.join(case_dir, "results_u.bp"), [u_n, v_n], engine="BP4"))
            vtx_s = stack.enter_context(VTXWriter(
                mesh0.comm, os.path.join(case_dir, "results_stress.bp"), post.fields, engine="BP4"))

            def write_all(tt):
                vtx_u.write(tt)
                vtx_s.write(tt)
        else:
            def write_all(tt):
                pass

        def record(tt):
            """Diagnostics of the accepted state (u_n, v_n, a_n)."""
            u.x.array[:] = u_n.x.array[:]          # forms f_el / stress use u
            u.x.scatter_forward()
            post.update()
            sig = post.value_at(target_node, comm=comm)
            sig = float("nan") if sig is None else float(sig)
            z_com = asm(f_z) / m_q
            v_com = asm(f_vz) / m_q
            a_com = asm(f_az) / m_q
            Fc = 4.0 * m_q * (a_com + G)           # net non-gravity force on the full ball
            Ek, Ee = asm(f_kin), asm(f_el)
            Eg = G * m_q * z_com
            row = (tt, z_com, v_com, gap_now(), Fc, sig, 4 * Ek, 4 * Ee, 4 * Eg)
            hist.append(row)
            if comm.rank == 0:
                with open(csv_path, "a") as f:
                    f.write(",".join(f"{x:.8e}" for x in row) + f",{row[6] + row[7] + row[8]:.8e}\n")
            return row

        row = record(0.0)
        write_all(0.0)
        gap = row[3]
        time_setup = MPI.Wtime() - t_start

        while t < T_END - 1e-12 and not aborted:
            dt = min(dt, DT_CONTACT if gap < GAP_FINE else DT_FREE, T_END - t)
            success = False
            for attempt in range(max_attempts):
                dt_c.value = dt
                # Newmark predictor (a_{n+1} ~ 0) as initial guess
                u.x.array[:] = (u_n.x.array + dt * v_n.x.array
                                + dt ** 2 * (0.5 - beta) * a_n.x.array)
                u.x.scatter_forward()
                x = u.x.petsc_vec.copy()
                try:
                    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    t0 = MPI.Wtime()
                    snes.solve(None, x)
                    time_snes += MPI.Wtime() - t0
                    reason = snes.getConvergedReason()
                    if reason <= 0:
                        raise RuntimeError(f"Newton diverged, reason={reason}")
                    x.copy(u.x.petsc_vec)
                    u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                              mode=PETSc.ScatterMode.FORWARD)
                    its = snes.getIterationNumber()
                    stats["newton"] += its
                    stats["steps"] += 1

                    # Newmark update of the state
                    a_next = ((u.x.array - u_n.x.array - dt * v_n.x.array) / (beta * dt ** 2)
                              - c_a * a_n.x.array)
                    v_next = v_n.x.array + dt * ((1.0 - gam) * a_n.x.array + gam * a_next)
                    a_n.x.array[:] = a_next
                    v_n.x.array[:] = v_next
                    u_n.x.array[:] = u.x.array
                    for fn in (a_n, v_n, u_n):
                        fn.x.scatter_forward()

                    t += dt
                    n += 1
                    row = record(t)
                    gap = row[3]
                    if n % WRITE_EVERY == 0:
                        write_all(t)
                    if n % 10 == 0 or gap < GAP_FINE:
                        emit(f"[p={fe_degree}, h={mesh_h:g}] step {n:5d} t={t:.5f} dt={dt:.2e} its={its} "
                             f"z={row[1]:.4f} v={row[2]:+.3f} gap={gap:.4f} Fc={row[4]:.3f} "
                             f"szz={row[5]:.2f} E={row[6] + row[7] + row[8]:.4f}")
                    dt = min(dt * 1.5, DT_FREE)
                    success = True
                except Exception as e:
                    emit(f"Step failed at t={t:.5f}, dt={dt:.3e}: {e}")
                    if VERBOSE and comm.rank == 0:
                        traceback.print_exc()
                    u.x.array[:] = u_n.x.array[:]
                    u.x.scatter_forward()
                    stats["failed"] += 1
                    dt = max(dt * 0.5, DT_MIN)
                    if dt == DT_MIN:
                        min_hits += 1
                    if min_hits >= 2:
                        emit(f"Minimum time step {DT_MIN:.1e} reached and still failing.")
                        aborted = True
                finally:
                    x.destroy()
                if success or aborted:
                    break
            if not success:
                emit(f"Time step {n + 1} failed. Final time reached: {t:.6f}")
                break

        write_all(t)
        emit(f"[p={fe_degree}, h={mesh_h:g}] Finished: {n} steps, t = {t:.5f}")
    finally:
        stack.close()
        snes.destroy(); J_L.destroy(); b.destroy()

    # ---- Summary quantities ----
    H = np.array(hist)
    i_in = int(np.argmin(H[:, 2]))                       # most negative ball velocity
    v_in = H[i_in, 2]
    v_out = H[i_in:, 2].max() if len(H) > i_in else float("nan")
    e_rest = v_out / abs(v_in) if v_in < 0 else float("nan")
    F_max = float(np.nanmax(H[:, 4])) if len(H) else float("nan")
    s_peak = float(H[np.nanargmin(H[:, 5]), 5]) if len(H) else float("nan")
    E_tot = H[:, 6] + H[:, 7] + H[:, 8]
    emit(f"    peak ball speed before impact: {abs(v_in):.3f} (free fall {v_imp:.3f}); "
         f"rebound speed {v_out:.3f}; restitution ~ {e_rest:.3f}")
    emit(f"    peak contact force (full ball) = {F_max:.3f} (Hertz {ref['F']:.3f}); "
         f"peak sigma_zz at target = {s_peak:.2f} (Hertz -p0 = {-ref['p0']:.2f}); "
         f"min gap = {H[:, 3].min():.4f}")
    emit(f"    total energy drift (excl. regularisation): {E_tot[0]:.4f} -> {E_tot[-1]:.4f}")

    if comm.rank == 0:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 2, figsize=(11, 8))
            ax[0, 0].plot(H[:, 0], H[:, 1] - (GEO.zc), label="ball COM height - initial")
            ax[0, 0].plot(H[:, 0], H[:, 3], label="gap below pole")
            ax[0, 0].set(xlabel="t", ylabel="height", title="Trajectory"); ax[0, 0].legend()
            ax[0, 1].plot(H[:, 0], H[:, 2]); ax[0, 1].set(xlabel="t", ylabel="v_z (COM)", title="Velocity")
            ax[1, 0].plot(H[:, 0], H[:, 4], label="F_contact (full ball)")
            ax[1, 0].axhline(ref["F"], ls="--", c="k", label="Hertz F_max")
            ax[1, 0].set(xlabel="t", ylabel="force", title="Contact force"); ax[1, 0].legend()
            ax[1, 1].plot(H[:, 0], H[:, 6], label="kinetic")
            ax[1, 1].plot(H[:, 0], H[:, 7], label="elastic")
            ax[1, 1].plot(H[:, 0], E_tot - E_tot[0], label="total - E0")
            ax[1, 1].set(xlabel="t", ylabel="energy", title="Energy"); ax[1, 1].legend()
            for a_ in ax.ravel():
                a_.grid(True, ls="--", alpha=0.5)
            fig.tight_layout()
            fig.savefig(os.path.join(case_dir, "trajectory.png"), dpi=200)
            plt.close(fig)
        except Exception as e:
            print(f"[trajectory plot skipped: {e}]")

    time_total = comm.allreduce(MPI.Wtime() - t_start, op=MPI.MAX)
    return {
        "fe_degree": fe_degree, "mesh": mesh0, "material_tags": material_tags, "mesh_h": mesh_h,
        "k_lulu_value": k_lulu, "k_huhu_value": k_huhu,
        "num_dofs": int(num_dofs), "num_cells": int(num_cells),
        "n_steps": n, "t_final": t, "newton_iterations": stats["newton"],
        "failed_attempts": stats["failed"],
        "avg_newton_per_step": stats["newton"] / n if n else float("nan"),
        "restitution": e_rest, "F_max": F_max, "sigma_zz_peak": s_peak, "hertz": ref,
        "energy_bodies": float(E_tot[-1]), "potential": float(E_tot[-1]),
        "time_total": time_total, "time_snes": time_snes,
        "history": H,
    }


# =============================================================================
if __name__ == "__main__":
    if comm.rank == 0:
        print_hardware_info()
    emit("\n" + "=" * 80 + f"\n RUN: dynamic bounce, reg={reg_method} {k_reg}, material={MATERIAL_MODEL}, "
         f"g={G}, gap0={GAP0}, T_END={T_END}, rho_ball={RHO_BALL}\n" + "=" * 80)

    cases = [(p, h) for p in p_values for h in h_values]
    sweep_results = []
    for degree, mesh_h in cases:
        res = run_case(mesh_h=mesh_h, fe_degree=degree, k_huhu=k_reg, k_lulu=k_reg,
                       write_outputs=True, comm=comm)
        sweep_results.append(res)
        _bk, _case_log = _case_log, None
        emit(f"p={degree} h={mesh_h:g}: steps={res['n_steps']}, newton={res['newton_iterations']}, "
             f"failed={res['failed_attempts']}, e~{res['restitution']:.3f}, "
             f"F_max={res['F_max']:.3f}, time={res['time_total']:.1f}s")
        _case_log = _bk

    _case_log = None
    emit("=" * 80)
    emit("Open results_u.bp / results_stress.bp in ParaView (VTX reader); "
         "trajectory.csv / trajectory.png per case.")
    if comm.rank == 0 and len(sweep_results) > 1:      # plotting.py is 2D/quad: guarded
        try:
            create_plots(case_name, sweep_dir, sweep_results)
        except Exception as e:
            print(f"[plots skipped: {e}]")