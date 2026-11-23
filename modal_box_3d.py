"""
3D clamped box ("modalbox"): MODAL (free-vibration) analysis, optionally about axially pre-compressed states,
plus the linear buckling analysis that defines the load levels.

  * right face (x = LX): fully clamped (u = 0)
  * left face  (x = 0) : prescribed axial end shortening (u_x = -s), u_y = u_z = 0   [static pre-load]
                         u = 0 on both ends for the vibration                         [dynamic BC]
  * the driver (driver3d.solve_with_load_stepping) needs a NEGATIVE target: load parameter s in [0, -delta]

Workflow
  Stage 1  linear buckling (as in stab_box_3d.py) -> P_cr and the linear critical end shortening delta_cr.
           Also prints the analytic clamped-clamped Euler load (4 pi^2 E I / L^2) as a cross-check.
  Stage 2  modal analysis. For every fraction f in LOAD_FRACS:
             - nonlinear equilibrium at end shortening f*delta_cr (f=0: undeformed state)
             - tangent stiffness K_T(u_f) (includes the geometric stiffness of the axial pre-load)
             - mass matrix M (solid density RHO, third medium RHO*RHO_MED_RATIO)
             - SLEPc shift-and-invert:  K_T phi = omega^2 M phi,   f = omega / (2 pi)
           Frequencies go to frequencies.csv, mode shapes to modes_f<frac>.bp (scaled to max |u| = 1).
           As the pre-load approaches the buckling load the lowest frequency drops towards zero.
  Stage 3  (NONLINEAR=1, default 0) displacement-controlled run to DISP_FACTOR*delta_cr with imperfection.

Environment variables
  RHO 1.0 | RHO_MED_RATIO 1e-8 | NMODES 10 | LOAD_FRACS "0:0.5:0.9" (":" , ";" or "," separated) | MODAL_SHIFT 0.0
  NEV 6 | NONLINEAR 0 | IMP_FRAC 1e-3 | DISP_FACTOR 2.0 | DISP_MAX | FORCE
  MU 10, K 20 | LAYER, LAYER_Z, OUTER_BC | ZBC must be 'free'
"""
import os

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

from dolfinx import default_scalar_type
from dolfinx.io import VTXWriter
from dolfinx.mesh import locate_entities, meshtags
from dolfinx.fem import (Constant, Function, dirichletbc, functionspace,
                         assemble_scalar, form, locate_dofs_geometrical)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_vector
from ufl import Measure, TestFunction, TrialFunction, derivative, inner

from methods import make_material, huhu_lulu_regularization, huhu_regularization, young_poisson
from functional_3d import kinematics_3d, psi_3d, psi_2_3d
from driver3d import (make_box_mesh, z_support_bcs,
                      solve_with_load_stepping, sweep_main)

MATERIAL_MODEL = os.environ.get("MATERIAL_MODEL", "hartmann_neff")
CELL_TYPE = os.environ.get("CELL_TYPE", "hexahedron")
ZBC = os.environ.get("ZBC", "free")
NZ = int(os.environ.get("NZ", "8"))
WRITE_EVERY = int(os.environ.get("WRITE_EVERY", "10"))
NEV = int(os.environ.get("NEV", "6"))
NONLINEAR = os.environ.get("NONLINEAR", "0") == "1"
IMP_FRAC = float(os.environ.get("IMP_FRAC", "1e-3"))
DISP_FACTOR = float(os.environ.get("DISP_FACTOR", "2.0"))
DISP_MAX_ENV = float(os.environ["DISP_MAX"]) if "DISP_MAX" in os.environ else None
FORCE_ENV = float(os.environ["FORCE"]) if "FORCE" in os.environ else None
RHO = float(os.environ.get("RHO", "1.0"))
RHO_MED_RATIO = float(os.environ.get("RHO_MED_RATIO", "1e-8"))
NMODES = int(os.environ.get("NMODES", "10"))
LOAD_FRACS = [float(s) for s in os.environ.get("LOAD_FRACS", "0:0.5:0.9").replace(",", ":").replace(";", ":").split(":")
              if s.strip()]
MODAL_SHIFT = float(os.environ.get("MODAL_SHIFT", "0.0"))
LAYER = os.environ.get("LAYER", "1") == "1"
LAYER_Z = LAYER and os.environ.get("LAYER_Z", "1") == "1"
OUTER_BC = os.environ.get("OUTER_BC", "free")
if ZBC == "plane_strain":
    raise ValueError("modalbox is a full 3D problem: use ZBC=free")
LX, LY = 2.0, 0.5
DEPTH = float(os.environ["DEPTH"]) if "DEPTH" in os.environ else LY

MU = float(os.environ.get("MU", "10.0"))
K = float(os.environ.get("K", "20.0"))
GAMMA = 1.0e-7
PRM0 = dict(Nsteps=100, Nsteps_min=10, Nsteps_max=800, inc=1.5, inc_late=1.1,
            dec=0.5, max_attempts=10,
            max_it=int(os.environ.get("NEWTON_MAXIT", "15")),
            atol=float(os.environ.get("NEWTON_ATOL", "1e-12")),
            rtol=float(os.environ.get("NEWTON_RTOL", "1e-9")),
            stol=float(os.environ.get("NEWTON_STOL", "1e-10")),
            linesearch="bt")


class AxialDisplacement:
    """Drop-in for the `disp` Constant of the load-stepping driver (load parameter s <= 0).

    `.value = s` prescribes u = (-s, 0, 0) on the left face and scales the lateral imperfection
    body force with (-s)/s_max.
    """

    def __init__(self, g, face_dofs, imp):
        self.g, self.dofs, self.imp = g, face_dofs, imp
        self.imp_total, self.s_max = 0.0, 1.0
        self._s = 0.0
        self.value = 0.0

    @property
    def value(self):
        return self._s

    @value.setter
    def value(self, s):
        self._s = float(np.asarray(s).ravel()[0])
        gv = self.g.x.array.reshape(-1, 3)
        gv[self.dofs, 0] = -self._s
        gv[self.dofs, 1] = 0.0
        gv[self.dofs, 2] = 0.0
        self.g.x.scatter_forward()
        self.imp.value = self.imp_total * (-self._s) / self.s_max


def run_case(Nx, Ny, fe_degree, reg_method, k_reg, out_dir, emit, verbose=False):
    comm = MPI.COMM_WORLD
    t_start = MPI.Wtime()
    if fe_degree < 2:
        raise ValueError("Hessian regularisation needs p >= 2")
    if Nx % 20 or Ny % 5:
        raise ValueError("monolith needs Nx % 20 == 0 and Ny % 5 == 0 so that the 0.1-wide "
                         "walls coincide with mesh lines")

    # ---- mesh ----
    nl = 1 if LAYER else 0
    nlz = 1 if LAYER_Z else 0
    hx, hy, hz = LX / Nx, LY / Ny, DEPTH / NZ
    ox, oy, oz = nl * hx, nl * hy, nlz * hz
    mesh0 = make_box_mesh(comm, LX + 2 * ox, LY + 2 * oy, DEPTH + 2 * oz,
                          Nx + 2 * nl, Ny + 2 * nl, NZ + 2 * nlz, CELL_TYPE)
    mesh0.geometry.x[:, :] -= np.array([ox, oy, oz])
    xmin, xmax, ymin, ymax = -ox, LX + ox, -oy, LY + oy
    zmin, zmax = -oz, DEPTH + oz

    tdim = mesh0.topology.dim
    fdim = tdim - 1
    mesh0.topology.create_connectivity(fdim, tdim)
    num_local_cells = mesh0.topology.index_map(tdim).size_local
    h = comm.allreduce(float(np.min(mesh0.h(tdim, np.arange(num_local_cells)))), op=MPI.MIN)

    Vu = functionspace(mesh0, ("Lagrange", fe_degree, (mesh0.geometry.dim,)))
    v = TestFunction(Vu)
    du = TrialFunction(Vu)
    u = Function(Vu, name="displacement")
    bs = Vu.dofmap.index_map_bs
    nown = Vu.dofmap.index_map.size_local
    nloc = nown * bs

    # ---- material tagging ----
    tol = 1e-8
    imap = mesh0.topology.index_map(tdim)
    marker = np.full(imap.size_local + imap.num_ghosts, 3, dtype=np.int32)
    in_box = lambda x: ((x[0] >= -tol) & (x[0] <= LX + tol) & (x[1] >= -tol) & (x[1] <= LY + tol)
                        & (x[2] >= -tol) & (x[2] <= DEPTH + tol))
    in_wall = lambda x: ((x[0] <= 0.1 + tol) | (x[0] >= LX - 0.1 - tol)
                         | (x[1] <= 0.1 + tol) | (x[1] >= 0.4 - tol))
    frame = locate_entities(mesh0, tdim, lambda x: in_box(x) & in_wall(x))
    marker[frame] = 1
    material_tags = meshtags(mesh0, tdim, np.arange(num_local_cells, dtype=np.int32),
                             marker[:num_local_cells])

    metadata = {"quadrature_degree": 2 * fe_degree + 2}
    dx = Measure("dx", domain=mesh0, subdomain_data=material_tags, metadata=metadata)
    vol = comm.allreduce(assemble_scalar(form(Constant(mesh0, default_scalar_type(1.0)) * dx(1))),
                         op=MPI.SUM)

    on_left = lambda x: (np.isclose(x[0], 0.0) & (x[1] >= -tol) & (x[1] <= LY + tol)
                         & (x[2] >= -tol) & (x[2] <= DEPTH + tol))
    on_right = lambda x: (np.isclose(x[0], LX) & (x[1] >= -tol) & (x[1] <= LY + tol)
                          & (x[2] >= -tol) & (x[2] <= DEPTH + tol))

    # ---- boundary conditions ----
    right_dofs = locate_dofs_geometrical(Vu, on_right)
    bcr = dirichletbc(np.zeros(3, dtype=default_scalar_type), right_dofs, Vu)

    # (a) displacement-controlled left face (static pre-load)
    left_dofs = locate_dofs_geometrical(Vu, on_left)
    g = Function(Vu, name="left_face_displacement")
    imp = Constant(mesh0, default_scalar_type(0.0))
    disp = AxialDisplacement(g, left_dofs, imp)
    bcl = dirichletbc(g, left_dofs)

    # (b) guided left face for the buckling eigenproblem: u_y = u_z = 0, u_x free
    guided = []
    for comp in (1, 2):
        Vc, _ = Vu.sub(comp).collapse()
        zero = Function(Vc)
        zero.x.array[:] = 0.0
        dofs = locate_dofs_geometrical((Vu.sub(comp), Vc), on_left)
        guided.append(dirichletbc(zero, dofs, Vu.sub(comp)))

    # (c) vibration: both ends fixed (homogeneous version of the displacement-controlled test)
    bc_left0 = dirichletbc(np.zeros(3, dtype=default_scalar_type), left_dofs, Vu)

    extra = list(z_support_bcs(mesh0, Vu, DEPTH, ZBC))
    if LAYER and OUTER_BC == "fixed":
        outer = lambda x: (np.isclose(x[0], xmin) | np.isclose(x[0], xmax)
                           | np.isclose(x[1], ymin) | np.isclose(x[1], ymax)
                           | np.isclose(x[2], zmin) | np.isclose(x[2], zmax))
        extra.append(dirichletbc(np.zeros(3, dtype=default_scalar_type),
                                 locate_dofs_geometrical(Vu, outer), Vu))
    bcs_nl = [bcl, bcr] + extra
    bcs_eig = guided + [bcr] + extra
    bcs_mod = [bc_left0, bcr] + extra

    # ---- energy ----
    mat = make_material(MU, K, MATERIAL_MODEL)
    kin = kinematics_3d(u)
    Pi_body = psi_3d(mat, kin) * dx(1)
    Pi_m = GAMMA * psi_2_3d(mat, kin) * dx(3)
    k = default_scalar_type(k_reg)
    R_reg = (huhu_lulu_regularization(u, v, dx, k, k, 3) if reg_method == "huhu_lulu"
             else huhu_regularization(u, v, dx, k, 3))
    L_int = derivative(Pi_body + Pi_m, u, v) + R_reg            # internal force vector
    L = L_int - imp * v[1] * dx(1)                              # + lateral imperfection load
    J_form = form(derivative(L_int, u, du))
    L_int_form = form(L_int)
    M_form = form(RHO * inner(du, v) * dx(1) + RHO * RHO_MED_RATIO * inner(du, v) * dx(3))

    dof_count = Vu.dofmap.index_map.size_global * bs
    num_cells = mesh0.topology.index_map(tdim).size_global
    emit(f"--- modalbox p={fe_degree}, {Nx}x{Ny}x{NZ}(+layer) {CELL_TYPE}, {num_cells} cells, "
         f"{dof_count} dofs, {reg_method} k={k_reg:.1e}, {MATERIAL_MODEL}, mu={MU:g}, K={K:g}, "
         f"rho={RHO:g} (medium x{RHO_MED_RATIO:g}), ZBC={ZBC}, depth={DEPTH:g}, "
         f"layer={LAYER}(z={LAYER_Z}), outer={OUTER_BC} ---")
    time_setup = MPI.Wtime() - t_start

    own_left = left_dofs[left_dofs < nown]

    def reaction():
        """axial reaction on the left face = - sum of internal-force x-components at its nodes"""
        f = assemble_vector(L_int_form)
        f.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fa = f.array_r.reshape(-1, bs)
        loc = float(fa[own_left, 0].sum()) if len(own_left) else 0.0
        f.destroy()
        return -comm.allreduce(loc, op=MPI.SUM)

    def lu_ksp(A):
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        return ksp

    # analytic cross-checks (thin-wall section: two walls of thickness 0.1, full depth)
    E_mod, _ = young_poisson(mat)
    wall = 0.1
    I_z_def = 2 * (DEPTH * wall ** 3 / 12 + wall * DEPTH * ((LY - wall) / 2) ** 2)   # deflection in y
    I_y_def = 2 * (wall * DEPTH ** 3 / 12)                                           # deflection in z
    I_min = min(I_z_def, I_y_def)
    A_sol = 2 * wall * DEPTH
    p_euler = 4 * np.pi ** 2 * E_mod * I_min / LX ** 2            # clamped-clamped column
    f1_est = 22.373 / (2 * np.pi) * np.sqrt(E_mod * I_min / (RHO * A_sol * LX ** 4))
    emit(f"analytic (E={E_mod:.4g}, I_min={I_min:.4e}): Euler P_E = {p_euler:.4g}, "
         f"first bending frequency (clamped-clamped, unloaded) f1 = {f1_est:.4g}")

    # =====================================================================================
    # Stage 1: linear buckling   K_G phi = mu K_E phi,   P_cr = -P0/mu
    # =====================================================================================
    t_buck = MPI.Wtime()
    u.x.array[:] = 0.0
    u.x.scatter_forward()
    KE = assemble_matrix(J_form, bcs=bcs_eig)                   # u_x free on left face
    KE.assemble()

    delta0 = 1e-3 * LX
    disp.value = -delta0
    A_nl = assemble_matrix(J_form, bcs=bcs_nl)
    A_nl.assemble()
    b = create_vector(Vu)
    b.set(0.0)
    apply_lifting(b, [J_form], [bcs_nl])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs_nl)

    ksp = lu_ksp(A_nl)
    x = A_nl.createVecRight()
    ksp.solve(b, x)
    u.x.array[:nloc] = x.array_r
    u.x.scatter_forward()
    P0 = reaction()
    emit(f"pre-stress: end shortening {delta0:.3e} -> reaction P0 = {P0:.6e}")
    ksp.destroy(); A_nl.destroy(); b.destroy(); x.destroy()

    KT = assemble_matrix(J_form, bcs=bcs_eig)
    KT.assemble()
    KG = KT.copy()
    KG.axpy(-1.0, KE, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)

    eps = SLEPc.EPS().create(comm)
    eps.setOperators(KG, KE)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    eps.setDimensions(max(NEV, 1))
    eps.setTolerances(1e-8, 500)
    st = eps.getST()
    st.getKSP().setType("preonly")
    st.getKSP().getPC().setType("lu")
    st.getKSP().getPC().setFactorSolverType("mumps")
    eps.setFromOptions()
    eps.solve()

    nconv = eps.getConverged()
    modes = []
    vr = KE.createVecRight()
    mode = Function(Vu, name="buckling_mode")
    mode_writer = VTXWriter(comm, os.path.join(out_dir, "buckling_modes.bp"), [mode], engine="BP4")
    try:
        for i in range(nconv):
            mu_i = float(np.real(eps.getEigenpair(i, vr)))
            if mu_i < 0.0:
                modes.append(-P0 / mu_i)
                mode.x.array[:nloc] = vr.array_r
                mode.x.scatter_forward()
                mode_writer.write(float(len(modes)))
    finally:
        mode_writer.close()
    eps.destroy(); KE.destroy(); KT.destroy(); KG.destroy(); vr.destroy()
    if not modes:
        raise RuntimeError("linear buckling: no negative eigenvalue found (check the pre-stress sign)")
    modes.sort()
    p_cr = modes[0]
    delta_cr = delta0 * p_cr / P0
    emit(f"linear buckling: P_cr = {', '.join(f'{p:.6g}' for p in modes[:NEV])}  "
         f"(P_cr,1 / P_Euler = {p_cr / p_euler:.4g}, [{MPI.Wtime() - t_buck:.1f} s])")
    emit(f"linear buckling: critical end shortening delta_cr = {delta_cr:.6g}")

    # =====================================================================================
    # Stage 2: modal analysis about pre-compressed states
    # =====================================================================================
    t_modal = MPI.Wtime()
    M = assemble_matrix(M_form, bcs=bcs_mod, diagonal=0.0)      # zero rows at constrained dofs
    M.assemble()
    mode_u = Function(Vu, name="vibration_mode")
    tot = dict(steps=0, newton=0, failed=0, ok=0, t_snes=0.0)
    freq_rows, freqs = [], {}
    last = dict(current=0.0, target=-1.0)

    for frac in LOAD_FRACS:
        delta_f = frac * delta_cr
        u.x.array[:] = 0.0
        u.x.scatter_forward()
        disp.value = 0.0
        disp.imp_total = 0.0                                    # perfect geometry for the vibration
        if delta_f > 0.0:
            disp.s_max = delta_f
            prm = dict(PRM0, crit=-0.8 * delta_f)
            s = solve_with_load_stepping(L, u, bcs_nl, disp, -delta_f, prm,
                                         emit=emit, verbose=verbose)
            tot["steps"] += s["load_steps"]; tot["newton"] += s["newton_iterations"]
            tot["failed"] += s["failed_attempts"]; tot["ok"] += s["successful_attempts"]
            tot["t_snes"] += s["time_snes"]
            last = dict(current=s["current"], target=-delta_f)
            if abs(s["current"] + delta_f) > 1e-8 * delta_f:
                emit(f"modal: equilibrium at f={frac:g} (shortening {delta_f:.5g}) NOT reached "
                     f"(stopped at {-s['current']:.5g}); skipping")
                continue
        p_f = reaction()

        Kt = assemble_matrix(J_form, bcs=bcs_mod)
        Kt.assemble()
        eps = SLEPc.EPS().create(comm)
        eps.setOperators(Kt, M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eps.setTarget(MODAL_SHIFT)
        eps.setDimensions(NMODES)
        eps.setTolerances(1e-8, 500)
        stm = eps.getST()
        stm.setType(SLEPc.ST.Type.SINVERT)
        stm.getKSP().setType("preonly")
        stm.getKSP().getPC().setType("lu")
        stm.getKSP().getPC().setFactorSolverType("mumps")
        eps.setFromOptions()
        eps.solve()

        nc = eps.getConverged()
        lam = np.array([float(np.real(eps.getEigenvalue(i))) for i in range(nc)])
        order = np.argsort(lam)
        vec = Kt.createVecRight()
        label = f"{frac:g}".replace(".", "p")
        writer_m = VTXWriter(comm, os.path.join(out_dir, f"modes_f{label}.bp"), [mode_u], engine="BP4")
        fl = []
        try:
            for rank_i, i in enumerate(order, start=1):
                eps.getEigenpair(int(i), vec)
                arr = vec.array_r
                mx = comm.allreduce(float(np.max(np.abs(arr))) if len(arr) else 0.0, op=MPI.MAX)
                mode_u.x.array[:nloc] = arr / mx if mx > 0 else arr
                mode_u.x.scatter_forward()
                writer_m.write(float(rank_i))
                w2 = lam[i]
                f_hz = np.sqrt(w2) / (2 * np.pi) if w2 > 0 else float("nan")
                fl.append(f_hz)
                freq_rows.append((frac, -disp.value, p_f, rank_i, w2, f_hz))
        finally:
            writer_m.close()
        freqs[frac] = fl
        emit(f"modal f={frac:g}: end shortening {-disp.value:.5g}, axial force {p_f:.5g} "
             f"({p_f / p_cr * 100:.1f}% of P_cr,lin); frequencies = "
             + ", ".join(f"{x:.5g}" for x in fl[:NMODES]))
        eps.destroy(); Kt.destroy(); vec.destroy()
    M.destroy()
    if comm.rank == 0 and freq_rows:
        np.savetxt(os.path.join(out_dir, "frequencies.csv"), np.array(freq_rows), delimiter=",",
                   header="fraction_of_delta_cr,end_shortening,axial_force,mode,omega2,frequency",
                   comments="")
    emit(f"modal analysis done [{MPI.Wtime() - t_modal:.1f} s]")

    # =====================================================================================
    # Stage 3 (optional): nonlinear displacement control with imperfection
    # =====================================================================================
    stats = dict(load_steps=0, current=0.0, time_snes=0.0, newton_iterations=0,
                 failed_attempts=0, successful_attempts=0)
    curve = []
    TARGET = last["target"]
    if NONLINEAR:
        dmax = DISP_MAX_ENV if DISP_MAX_ENV is not None else DISP_FACTOR * delta_cr
        force_ref = FORCE_ENV if FORCE_ENV is not None else p_cr
        TARGET = -dmax
        prm = dict(PRM0, crit=0.8 * TARGET)
        disp.s_max = dmax
        disp.imp_total = IMP_FRAC * force_ref / vol
        u.x.array[:] = 0.0
        u.x.scatter_forward()
        disp.value = 0.0
        emit(f"nonlinear stage: end shortening 0 -> {dmax:.6g}, imperfection = "
             f"{IMP_FRAC:g} * {force_ref:.6g}")

        def on_step(n, d):
            curve.append((-disp.value, reaction()))
            if n % WRITE_EVERY == 0:
                writer.write(float(n))

        writer = VTXWriter(comm, os.path.join(out_dir, "results_u.bp"), [u], engine="BP4")
        try:
            writer.write(0.0)
            stats = solve_with_load_stepping(L, u, bcs_nl, disp, TARGET, prm,
                                             emit=emit, verbose=verbose, on_step=on_step)
            writer.write(float(stats["load_steps"]) + 0.5)
        finally:
            writer.close()
        if comm.rank == 0 and curve:
            np.savetxt(os.path.join(out_dir, "force_displacement.csv"), np.array(curve),
                       delimiter=",", header="end_shortening,axial_force", comments="")
        if curve:
            pmax = max(p for _, p in curve)
            emit(f"nonlinear: max axial force {pmax:.6g} (P_cr,lin = {p_cr:.6g}, "
                 f"ratio {pmax / p_cr:.4g}); final force {curve[-1][1]:.6g}")
        tot["steps"] += stats["load_steps"]; tot["newton"] += stats["newton_iterations"]
        tot["failed"] += stats["failed_attempts"]; tot["ok"] += stats["successful_attempts"]
        tot["t_snes"] += stats["time_snes"]
        last = dict(current=stats["current"], target=TARGET)

    potential = comm.allreduce(assemble_scalar(form(Pi_body + Pi_m)), op=MPI.SUM)
    steps = tot["steps"]
    return {
        "fe_degree": fe_degree, "Nx": Nx, "Ny": Ny, "Nz": NZ, "mesh_h": h,
        "dof": int(dof_count), "num_cells": int(num_cells),
        "final_displacement": last["current"],
        "P_cr_linear": p_cr, "P_cr_all": modes, "delta_cr": delta_cr, "frequencies": freqs,
        "n_steps": steps, "potential": potential,
        "time_total": comm.allreduce(MPI.Wtime() - t_start, op=MPI.MAX),
        "time_setup": comm.allreduce(time_setup, op=MPI.MAX),
        "time_snes": comm.allreduce(tot["t_snes"], op=MPI.MAX),
        "newton_iterations": tot["newton"],
        "avg_newton_per_step": tot["newton"] / steps if steps else 0.0,
        "failed_attempts": tot["failed"],
        "successful_attempts": tot["ok"],
        "disp_pct": (last["current"] / last["target"] * 100.0) if last["target"] else 0.0,
        "u": u, "Vu": Vu, "mesh": mesh0, "material_tags": material_tags,
    }


if __name__ == "__main__":
    sweep_main("modalbox3d", run_case, default_meshes="20x10,40x20,60x30")