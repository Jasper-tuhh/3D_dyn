"""
driver3d.py -- shared pieces of the 3D benchmarks (cbox_3d.py, monolith_3d.py).

* make_box_mesh      : 2D rectangle  ->  extruded box (hexahedra or tetrahedra)
* dofs_on_edges      : the 2D "vertex" BCs become BCs on edges parallel to z
* z_support_bcs      : optional u_z = 0 on both z-faces (exact plane-strain equivalent)
* solve_with_load_stepping : adaptive displacement control (same logic as the 2D scripts,
                       but one SNES / one BC object for the whole run: the prescribed
                       displacement is a Constant that is updated in place)
* sweep_main         : loops over regularisation methods, degrees and meshes and writes
                       the same log table as the 2D scripts (plotting.from_log compatible)

Environment variables (all optional)
    REG_METHODS   huhu_lulu,huhu          P_VALUES  2        (p >= 2 is required)
    REG_VALUE     1e-5                    MESHES    NxxNy    e.g. 20x10,40x20  (2D definition)
    NZ            8   elements over the depth     DEPTH   extrusion depth (default: 2D width)
    ZBC           free | plane_strain     CELL_TYPE hexahedron | tetrahedron
    MATERIAL_MODEL, WRITE_EVERY (VTX output every n load steps, default 10), VERBOSE
"""
import os
import traceback

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import default_scalar_type
from dolfinx.mesh import CellType, create_box, locate_entities_boundary
from dolfinx.fem import Constant, Function, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import create_matrix, create_vector

from functional import emit as _emit, print_hardware_info, NonlinearPDE_SNESProblem
from plotting import create_plots, from_log


# ----------------------------------------------------------------------------
# configuration helpers
# ----------------------------------------------------------------------------
def env_list(name, default, cast=str):
    return [cast(x) for x in os.environ.get(name, default).split(",") if x.strip()]


def parse_meshes(s):
    return [tuple(int(v) for v in item.lower().split("x")) for item in s.split(",") if item.strip()]


# ----------------------------------------------------------------------------
# mesh + boundary conditions
# ----------------------------------------------------------------------------
def make_box_mesh(comm, Lx, Ly, depth, Nx, Ny, Nz, cell_type="hexahedron"):
    return create_box(comm, [[0.0, 0.0, 0.0], [Lx, Ly, depth]], [Nx, Ny, Nz],
                      getattr(CellType, cell_type))


def dofs_on_edges(mesh, space, marker):
    """Dofs of `space` (a sub-space or the full space) on boundary edges selected by marker(x)."""
    tdim = mesh.topology.dim
    mesh.topology.create_entities(1)
    mesh.topology.create_connectivity(1, tdim)
    edges = locate_entities_boundary(mesh, 1, marker)
    return locate_dofs_topological(space, 1, edges)


def z_support_bcs(mesh, V, depth, mode):
    """mode='plane_strain': u_z = 0 on z=0 and z=depth  (3D solution == extruded 2D solution)."""
    if mode != "plane_strain":
        return []
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    facets = locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[2], 0.0) | np.isclose(x[2], depth))
    zero = Constant(mesh, default_scalar_type(0))
    return [dirichletbc(zero, locate_dofs_topological(V.sub(2), fdim, facets), V.sub(2))]


# ----------------------------------------------------------------------------
# adaptive displacement-controlled solve
# ----------------------------------------------------------------------------
def solve_with_load_stepping(L, u, bcs, disp, target, prm, emit=None, on_step=None, verbose=False):
    """disp: Constant used in a Dirichlet BC (updated in place). target < 0.
    SNES reasons: -5 max iterations, -6 line search failed, -3 linear solve failed, -4 NaN.
    prm keys: Nsteps, Nsteps_min, Nsteps_max, crit, inc, inc_late, dec, max_attempts,
              max_it, atol, rtol, stol, linesearch"""
    V = u.function_space
    comm = V.mesh.comm
    u_prev = Function(V)
    problem = NonlinearPDE_SNESProblem(L, u, bcs)
    b = create_vector(V)
    J_L = create_matrix(problem.a)
    snes = PETSc.SNES().create(comm)
    snes.setOptionsPrefix("bench_")
    snes.setFunction(problem.F, b)
    snes.setJacobian(problem.J, J_L)
    snes.setTolerances(atol=prm["atol"], rtol=prm["rtol"], stol=prm["stol"], max_it=prm["max_it"])
    snes.getKSP().setType("preonly")
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")
    opts = PETSc.Options()
    opts["bench_snes_linesearch_type"] = prm["linesearch"]
    opts["bench_snes_max_linear_solve_fail"] = 20
    opts["bench_snes_linesearch_max_it"] = 20
    if verbose:
        opts["bench_snes_monitor"] = None
        opts["bench_snes_converged_reason"] = None
    snes.setFromOptions()

    stats = {"newton_iterations": 0, "load_steps": 0, "failed_attempts": 0,
             "successful_attempts": 0}
    time_snes = 0.0
    initial = abs(target) / prm["Nsteps"]
    min_step = abs(target) / prm["Nsteps_max"]
    max_step = abs(target) / prm["Nsteps_min"]
    step_size, inc = initial, prm["inc"]
    current, n, min_hits, aborted = 0.0, 0, 0, False
    say = emit if emit is not None else (lambda m: None)

    try:
        while current > target and not aborted:
            success = False
            for _ in range(prm["max_attempts"]):
                nxt = max(current - step_size, target)
                disp.value = nxt
                u_prev.x.array[:] = u.x.array[:]
                x = u.x.petsc_vec.copy()
                try:
                    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    t0 = MPI.Wtime()
                    snes.solve(None, x)
                    time_snes += MPI.Wtime() - t0
                    reason = snes.getConvergedReason()
                    if reason <= 0:
                        raise RuntimeError(f"Newton failed: reason={reason}, "
                                        f"its={snes.getIterationNumber()}, "
                                        f"|F|={snes.getFunctionNorm():.3e}")
                    x.copy(u.x.petsc_vec)
                    u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                              mode=PETSc.ScatterMode.FORWARD)
                    its = snes.getIterationNumber()
                    stats["newton_iterations"] += its
                    stats["successful_attempts"] += 1
                    stats["load_steps"] += 1
                    n += 1
                    current = nxt
                    if current <= prm["crit"]:
                        inc = prm["inc_late"]
                    step_size = min(step_size * inc, max_step)
                    success = True
                    if verbose:
                        say(f"  step {n}: disp {current:.5f}, its={its}")
                    if on_step is not None:
                        on_step(n, current)
                except Exception as e:
                    say(f"Step failed at disp {current:.5f} -> {nxt:.5f}: {e}")
                    if verbose and comm.rank == 0:
                        traceback.print_exc()
                    u.x.array[:] = u_prev.x.array[:]
                    u.x.scatter_forward()
                    stats["failed_attempts"] += 1
                    step_size = max(step_size * prm["dec"], min_step)
                    if step_size == min_step:
                        min_hits += 1
                    if min_hits >= 2:
                        say(f"Minimum step size {min_step:.3e} reached and still failing.")
                        aborted = True
                finally:
                    x.destroy()
                if success or aborted:
                    break
            if not success:
                break
    finally:
        snes.destroy()
        J_L.destroy()
        b.destroy()
    stats.update(current=current, time_snes=time_snes)
    return stats


# ----------------------------------------------------------------------------
# sweep driver (same log layout as the 2D scripts)
# ----------------------------------------------------------------------------
def sweep_main(case, run_case, default_meshes, comm=MPI.COMM_WORLD):
    reg_methods = env_list("REG_METHODS", "huhu_lulu,huhu")
    degrees = env_list("P_VALUES", "2", int)
    meshes = parse_meshes(os.environ.get("MESHES", default_meshes))
    k_reg = float(os.environ.get("REG_VALUE", "1e-5"))
    verbose = os.environ.get("VERBOSE", "0") == "1"

    if comm.rank == 0:
        print_hardware_info()
    for reg_method in reg_methods:
        assert reg_method in ("huhu_lulu", "huhu"), reg_method
        file_name = f"{case}_{reg_method}"
        dir_name = os.path.join(".", file_name)
        if comm.rank == 0:
            os.makedirs(dir_name, exist_ok=True)
        comm.Barrier()
        log_path = os.path.join(dir_name, "terminal.txt")

        def emit(msg, _lp=log_path):
            _emit(_lp, msg, comm)

        _emit(log_path, f"{case}: reg_method={reg_method}, k={k_reg:.1e}, degrees={degrees}, "
                        f"meshes={meshes}", comm, mode="w")
        sweep_results = []
        for fe_degree in degrees:
            emit(f"\nPolynomial degree p = {fe_degree}")
            emit("-" * 100)
            emit(f"{'Mesh size (h)':<15} | {'Trgt Dsp [%]':<14} | {'Load steps':<12} | "
                 f"{'Newton iters':<14} | {'Avg. Newton/Step':<18} | {'Solver time [s]':<15}")
            emit("-" * 100)
            for (Nx, Ny) in meshes:
                out_dir = os.path.join(dir_name, f"p{fe_degree}_{Nx}x{Ny}")
                if comm.rank == 0:
                    os.makedirs(out_dir, exist_ok=True)
                comm.Barrier()
                res = run_case(Nx, Ny, fe_degree, reg_method, k_reg, out_dir, emit, verbose)
                sweep_results.append(res)
                emit(f"{res['mesh_h']:<15.3e} | {res['disp_pct']:<14.2f} | {res['n_steps']:<12d} | "
                     f"{res['newton_iterations']:<14d} | {res['avg_newton_per_step']:<18.2f} | "
                     f"{res['time_total']:<15.3f}")
                emit(f"   [{Nx}x{Ny}x{res['Nz']}, {res['num_cells']} cells, {res['dof']} dofs, "
                     f"potential = {res['potential']:.6e}]")
            emit("-" * 100)

        if comm.rank == 0:      # plotting.py was written for 2D: never let it kill a finished run
            try:
                create_plots(file_name, dir_name, sweep_results)
                from_log(log_path, file_name, dir_name)
            except Exception as e:
                print(f"[plots skipped: {e}]")