"""
3D Hertzian contact: elastic sphere (ball block) on an elastic substrate,
void replaced by a third medium (tag 3) with huhu/lulu regularisation.
Quarter model, hexahedral elements.  Extension of the 2D script.

Differences to the 2D script
  * mesh      : mesh3d.generate_mesh_3d (structured hexahedra, geometry order 1)
  * space     : vector P_p(3)
  * BCs       : u_z = 0 on the bottom, symmetry u_x = 0 (x=0), u_y = 0 (y=0)
                (replaces the single-node corner pin of the 2D model)
  * load      : traction (0, 0, -p) on the flat ball top (facet tag 11)
  * kinematics: functional_3d (full 3x3 F, no plane-strain embedding)
  * stresses  : functional_3d.StressPostProcessor3D (sigma_22 -> sigma_zz)
  * 2D-only plotting removed; results go to ParaView (.bp files)
"""
import os
import traceback
from contextlib import ExitStack

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from ufl import Measure, TestFunction, derivative, inner
from dolfinx import default_scalar_type
from dolfinx.io import VTXWriter
from dolfinx.fem import (Constant, Function, dirichletbc, functionspace,
                         locate_dofs_topological, assemble_scalar, form)
from dolfinx.fem.petsc import create_matrix, create_vector

from methods import (make_material, young_poisson,
                     huhu_lulu_regularization, huhu_regularization)
from functional import (emit as _emit, print_hardware_info, make_element,
                        NonlinearPDE_SNESProblem)
from functional_3d import (kinematics_3d, psi_3d, psi_2_3d, cauchy_stress_3d,
                           StressPostProcessor3D)
from plotting import create_plots, from_log
from mesh3d import (Geometry3D, generate_mesh_3d,
                    TAG_BOTTOM, TAG_LOAD, TAG_SYM_X, TAG_SYM_Y)

comm = MPI.COMM_WORLD

h_values = [float(x) for x in os.environ.get("H_VALUES", "0.025").split(",")]
p_values = [int(x) for x in os.environ.get("P_VALUES", "2").split(",")]

# ---- Material --------------------------------------------------------------
MATERIAL_MODEL = os.environ.get("MATERIAL_MODEL", "neo_hooke")

# ---- Elements (hexahedra) --------------------------------------------------
ELEMENT = dict(family="Lagrange", variant="gll_warped")
STRESS_ELEMENT = dict(degree=0, variant="gll_warped")
QUADRATURE = dict(rule="GLL", degree="auto")      # auto: 2p-1 (nodal/GLL quadrature)

# ---- Geometry: same dimensions as hertzian0_4.geo (sphere radius R = 1) ------
GEO = Geometry3D()          # gap 0.05, half-width 1.1, depth 2.0, top at 2.0, load patch +-0.25
N_GAP_LAYERS = 3

# ---- Regularisation --------------------------------------------------------
reg_methods = ["huhu_lulu", "huhu"]
reg_method = os.environ.get("REG_METHOD", "huhu_lulu")
assert reg_method in reg_methods, reg_method
reg_array = [1e-3]
k_huhu_value = reg_array[0]
k_lulu_value = reg_array[0]

# ---- Solver / load stepping --------------------------------------------------
linesearch = "bt"
# traction on the loaded patch (|x|,|y| <= w_load) of the ball top. Same value as in 2D;
# full-ball force = 4 * p * w_load^2 (= 150 here).
target_load = float(os.environ.get("TARGET_LOAD", "-600"))
max_it_newton = 10
Nsteps = 200; Nsteps_min = 25; Nsteps_max = 40000
increase_factor = 2.0; decrease_factor = 0.5
max_attempts = 10

# ---- Materials & misc --------------------------------------------------------
mu_c = default_scalar_type(35000 / 13); K_c = default_scalar_type(17500 / 3)     # ball
mu_e = default_scalar_type(700000 / 29); K_e = default_scalar_type(700000 / 3)   # substrate
mat_c = make_material(mu_c, K_c, MATERIAL_MODEL)
mat_e = make_material(mu_e, K_e, MATERIAL_MODEL)
gamma = default_scalar_type(1.3e-3)
# contact centre on the substrate surface (2D: (1.1, 2.0)); sigma_zz here ~ -p0 (Hertz).
# value_at() prefers substrate cells on the interface, as in 2D.
target_node = (0.0, 0.0, 0.0)


def hertz_sphere(force_total):
    """Small-strain Hertz sphere/half-space (reference only: the ball here is a block)."""
    E1, n1 = young_poisson(mat_c)
    E2, n2 = young_poisson(mat_e)
    Es = 1.0 / ((1 - n1 ** 2) / E1 + (1 - n2 ** 2) / E2)
    F = abs(force_total)
    a = (3 * F * GEO.R / (4 * Es)) ** (1 / 3)
    p0 = 3 * F / (2 * np.pi * a ** 2) if a > 0 else 0.0
    return a, p0, a ** 2 / GEO.R


# =============================================================================
# Cases and output folders
# =============================================================================

cases = [(p, h) for p in p_values for h in h_values]

sweep_name = f"hertz3d_sweep_{reg_method}_{MATERIAL_MODEL}"
sweep_dir = os.path.join(".", sweep_name)
case_name = f"hertz3d_kh{k_huhu_value:.1e}_kl{k_lulu_value:.1e}_{reg_method}"
sweep_log_path = os.path.join(sweep_dir, "sweep_terminal.txt")
_case_log = None

if comm.rank == 0:
    os.makedirs(sweep_dir, exist_ok=True)
    with open(sweep_log_path, "w") as f:
        f.write("3D Hertzian Contact Sweep Log\n" + "=" * 40 + "\n")
comm.Barrier()


def emit(msg):
    """Print + log (current case log and sweep log) on rank 0."""
    if _case_log is not None:
        _emit(_case_log, msg, comm)
    _emit(sweep_log_path, msg, comm, echo=_case_log is None)


def run_case(mesh_h, fe_degree, k_huhu, k_lulu, target_load=target_load,
             Nsteps=Nsteps, write_outputs=True, comm=MPI.COMM_WORLD):
    global _case_log
    t_start = MPI.Wtime()

    case_dir = os.path.join(sweep_dir, f"{case_name}_p{fe_degree}_h{mesh_h:g}")
    if comm.rank == 0:
        os.makedirs(case_dir, exist_ok=True)
        with open(os.path.join(case_dir, "terminal.txt"), "w") as f:
            f.write(f"3D Hertzian Contact: p={fe_degree}, h={mesh_h:g}, material={MATERIAL_MODEL}\n")
    comm.Barrier()
    _case_log = os.path.join(case_dir, "terminal.txt")

    stats = {"load_steps": 0, "newton_iterations": 0, "newton_iterations_failed": 0,
             "failed_attempts": 0, "successful_attempts": 0}
    time_snes = 0.0

    initial_step_size = abs(target_load) / Nsteps
    min_step_size = abs(target_load) / Nsteps_max
    max_step_size = abs(target_load) / Nsteps_min

    # ---- Mesh ----
    geo_order = 1
    mesh_data = generate_mesh_3d(mesh_h=mesh_h, comm=comm, geo=GEO, n_gap=N_GAP_LAYERS)
    mesh0 = mesh_data.mesh
    material_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags
    tdim = mesh0.topology.dim
    fdim = tdim - 1
    mesh0.topology.create_connectivity(fdim, tdim)
    mesh0.topology.create_connectivity(0, tdim)

    q_deg = 2 * fe_degree - 1 if QUADRATURE["degree"] == "auto" else int(QUADRATURE["degree"])
    metadata = {"quadrature_rule": "GLL", "quadrature_degree": q_deg}

    dx = Measure("dx", domain=mesh0, subdomain_data=material_tags, metadata=metadata)
    ds = Measure("ds", domain=mesh0, subdomain_data=facet_tags, metadata=metadata)

    # ---- Function space ----
    cell = mesh0.basix_cell()
    P_u = make_element(cell, fe_degree, shape=(mesh0.geometry.dim,), **ELEMENT)
    V = functionspace(mesh0, P_u)
    v = TestFunction(V)
    u = Function(V)
    u_prev = Function(V)

    # ---- Boundary conditions (quarter model) ----
    zero = Constant(mesh0, default_scalar_type(0))
    bcs_def = [(2, TAG_BOTTOM),    # roller support: u_z = 0
               (0, TAG_SYM_X),     # symmetry plane x = 0: u_x = 0
               (1, TAG_SYM_Y)]     # symmetry plane y = 0: u_y = 0
    bcu = [dirichletbc(zero, locate_dofs_topological(V.sub(c), fdim, facet_tags.find(tag)), V.sub(c))
           for c, tag in bcs_def]

    pload = Constant(mesh0, default_scalar_type((0.0, 0.0, 0.0)))

    # ---- Energy and residual ----
    kin = kinematics_3d(u)
    psi_e = psi_3d(mat_e, kin)
    psi_c = psi_3d(mat_c, kin)
    psi_m = gamma * psi_2_3d(mat_c, kin)

    Pi_body = psi_e * dx(1) + psi_c * dx(2) - inner(pload, u) * ds(TAG_LOAD)
    Pi_m = psi_m * dx(3)

    if reg_method == "huhu_lulu":
        R_reg = huhu_lulu_regularization(u, v, dx, k_huhu, k_lulu, 3)
    else:
        R_reg = huhu_regularization(u, v, dx, k_huhu, 3)
    L = derivative(Pi_body + Pi_m, u, v) + R_reg

    # ---- Stress output ----
    stress_el = make_element(cell, fe_degree, family="Lagrange",
                             variant=STRESS_ELEMENT["variant"], discontinuous=True)
    post = StressPostProcessor3D(mesh0, material_tags,
                                 cauchy_stress_3d(psi_e, kin), cauchy_stress_3d(psi_c, kin),
                                 cauchy_stress_3d(psi_m, kin), stress_el)

    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    num_cells = mesh0.topology.index_map(tdim).size_global
    emit(f"--- p={fe_degree}, h={mesh_h:g}: {num_cells} hex cells, {num_dofs} dofs, "
         f"geometry order {geo_order}, quadrature {metadata} ---")
    if fe_degree < 2:
        raise ValueError("Hessian-based regularisation needs p >= 2 (Q1 has no pure second derivatives).")

    # ---- SNES ----
    problem = NonlinearPDE_SNESProblem(L, u, bcs=bcu)
    b = create_vector(V)
    J_L = create_matrix(problem.a)
    snes = PETSc.SNES().create(comm)
    snes.setOptionsPrefix("hertz_")
    snes.setFunction(problem.F, b)
    snes.setJacobian(problem.J, J_L)
    snes.setTolerances(atol=1e-8, rtol=1e-9, stol=1e-10, max_it=max_it_newton)
    snes.getKSP().setType("preonly")
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")
    opts = PETSc.Options()
    opts["hertz_snes_linesearch_type"] = linesearch
    opts["hertz_snes_max_linear_solve_fail"] = 20
    opts["hertz_snes_linesearch_max_it"] = 20
    opts["hertz_snes_monitor"] = None
    opts["hertz_snes_converged_reason"] = None
    snes.setFromOptions()

    sigma_zz_at_node = None
    current_load = 0.0
    n = 0

    stack = ExitStack()
    try:
        if write_outputs:
            vtx_u = stack.enter_context(VTXWriter(
                mesh0.comm, os.path.join(case_dir, "results_u.bp"), [u], engine="BP4"))
            vtx_s = stack.enter_context(VTXWriter(
                mesh0.comm, os.path.join(case_dir, "results_stress.bp"), post.fields, engine="BP4"))

            def write_all(t):
                vtx_u.write(t)
                vtx_s.write(t)
        else:
            def write_all(t):
                pass

        time_setup = MPI.Wtime() - t_start

        step_size = initial_step_size
        min_hits = 0
        aborted = False

        while current_load > target_load and not aborted:
            success = False
            for attempt in range(1, max_attempts + 1):
                next_load = max(current_load - step_size, target_load)
                pload.value = np.array([0.0, 0.0, next_load], dtype=default_scalar_type)
                u_prev.x.array[:] = u.x.array[:]

                x = u.x.petsc_vec.copy()
                try:
                    emit(f"[p={fe_degree}, h={mesh_h:g}] step {n + 1}, "
                         f"load {current_load:.4f} -> {next_load:.4f}, {current_load/target_load*100:.2f}% of target load")
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
                    stats["newton_iterations"] += snes.getIterationNumber()
                    stats["successful_attempts"] += 1
                    stats["load_steps"] += 1
                    emit(f"  its={snes.getIterationNumber()}, |F|={snes.getFunctionNorm():.3e}")

                    post.update()
                    sigma_zz_at_node = post.value_at(target_node, comm=comm)

                    n += 1
                    current_load = next_load
                    step_size = min(step_size * increase_factor, max_step_size)
                    success = True

                except Exception as e:
                    emit(f"Step failed: {e}")
                    if comm.rank == 0:
                        traceback.print_exc()
                    u.x.array[:] = u_prev.x.array[:]
                    u.x.scatter_forward()
                    stats["failed_attempts"] += 1
                    stats["newton_iterations_failed"] += snes.getIterationNumber()
                    step_size = max(step_size * decrease_factor, min_step_size)
                    emit(f"Reducing step size to {step_size:.6f}")
                    if step_size == min_step_size:
                        min_hits += 1
                    if min_hits >= 2:
                        emit(f"Minimum step size {min_step_size} reached and still failing.")
                        aborted = True
                finally:
                    x.destroy()

                if success or aborted:
                    break

            if not success:
                emit(f"Step {n + 1} failed. Final load achieved: {current_load:.6f}")
                break

        write_all(float(n))
        emit(f"[p={fe_degree}, h={mesh_h:g}] Finished. Successful steps: {n}, "
             f"final load: {current_load:.6f}")
    finally:
        stack.close()
        snes.destroy()
        J_L.destroy()
        b.destroy()

    # ---- Post-run quantities (collective) ----
    energy_bodies = comm.allreduce(assemble_scalar(form(Pi_body)), op=MPI.SUM)
    targ_pct = (current_load / target_load) * 100.0 if target_load != 0 else 0.0
    avg_newton = (stats["newton_iterations"] / stats["load_steps"]
                  if stats["load_steps"] > 0 else float("nan"))
    time_total = comm.allreduce(MPI.Wtime() - t_start, op=MPI.MAX)
    time_setup = comm.allreduce(time_setup, op=MPI.MAX)
    time_snes = comm.allreduce(time_snes, op=MPI.MAX)

    # Hertz reference at the load actually reached (quarter model -> x4 for the full ball)
    force_total = 4.0 * abs(current_load) * GEO.w_load ** 2
    a_h, p0_h, delta_h = hertz_sphere(force_total)

    return {
        "fe_degree": fe_degree, "geometry_order": geo_order,
        "material_model": MATERIAL_MODEL, "element": dict(ELEMENT),
        "k_lulu_value": k_lulu, "k_huhu_value": k_huhu,
        "mesh": mesh0, "material_tags": material_tags, "mesh_h": mesh_h,
        "num_dofs": int(num_dofs), "num_cells": int(num_cells),
        "sigma_zz_at_target_node": (float(sigma_zz_at_node) if sigma_zz_at_node is not None else None),
        "sigma22_at_target_node": (float(sigma_zz_at_node) if sigma_zz_at_node is not None else None),  # key used by plotting.py
        "hertz_a": a_h, "hertz_p0": p0_h, "hertz_delta": delta_h, "force_total": force_total,
        "final_load_reached": float(current_load), "n_steps": int(n),
        "energy_bodies": float(energy_bodies), "potential": float(energy_bodies),
        "targ_pct": float(targ_pct),
        "time_total": time_total, "time_setup": time_setup, "time_snes": time_snes,
        "newton_iterations": stats["newton_iterations"], "avg_newton_per_step": avg_newton,
        "failed_attempts": stats["failed_attempts"],
        "successful_attempts": stats["successful_attempts"],
        "u": u, "Vu": V,
    }


# =============================================================================
if __name__ == "__main__":
    if comm.rank == 0:
        print_hardware_info()
    emit("\n" + "=" * 80 + f"\n RUN: reg_method={reg_method}, reg_value={k_huhu_value}, "
         f"material={MATERIAL_MODEL}, target traction={target_load}\n" + "=" * 80)

    sweep_results = []
    last_degree = None
    for degree, mesh_h in cases:
        res = run_case(mesh_h=mesh_h, fe_degree=degree, k_huhu=k_huhu_value,
                       k_lulu=k_lulu_value, target_load=target_load, Nsteps=Nsteps,
                       write_outputs=True, comm=comm)
        sweep_results.append(res)

        _case_log_backup, _case_log = _case_log, None     # table goes to the sweep log only
        if degree != last_degree:                         # parsed by plotting.from_log
            emit(f"Polynomial degree p = {degree}")
            last_degree = degree
        emit(f"{'h':<12} | {'target %':<10} | {'steps':<8} | {'newton':<8} | "
             f"{'avg newton':<10} | {'time [s]':<10}")
        emit(f"{res['mesh_h']:<12.5g} | {res['targ_pct']:<10.2f} | {res['n_steps']:<8d} | "
             f"{res['newton_iterations']:<8d} | {res['avg_newton_per_step']:<10.3f} | "
             f"{res['time_total']:<10.3f}")
        # label "sigma22" kept so plotting.from_log still parses it; value is sigma_zz (vertical)
        emit(f"   sigma22 at target node = {res['sigma_zz_at_target_node']}, "
             f"energy = {res['energy_bodies']:.3f}, dofs = {res['num_dofs']}")
        emit(f"   Hertz reference (F={res['force_total']:.3f}): a = {res['hertz_a']:.4f}, "
             f"p0 = {res['hertz_p0']:.2f}, delta = {res['hertz_delta']:.4f}")
        _case_log = _case_log_backup

    _case_log = None
    emit("=" * 80)
    emit("Open results_u.bp / results_stress.bp in ParaView (VTX reader).")
    if comm.rank == 0:      
        try:
            create_plots(case_name, sweep_dir, sweep_results)
            from_log(sweep_log_path, sweep_name, sweep_dir)
        except Exception as e:
            print(f"[plots skipped: {e}]")