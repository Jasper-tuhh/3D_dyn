"""Problem definition: mesh, regularisation and material models.

Switch the material with  make_material(mu, K, model="hartmann_neff" | "neo_hooke" | ...).
Add your own with register_material(...) at the bottom of this file.
All strain energies receive the kinematics namespace `kin`
(kin.J, kin.Ic, kin.Ib1, kin.Ib2, kin.F) built by functional.kinematics(u).
"""
import math

import numpy as np
import gmsh
from ufl import grad, div, inner, ln

# ============================================================================
# Mesh
# ============================================================================
def _build_geometry(mesh_h):
    """Re-creates hertzian0_4.geo in the currently-initialized gmsh model.
    Everything (including the local Ball-field refinement) scales off the
    single parameter `mesh_h`, exactly as in the original .geo file."""
    p0x, p0y = 0.0, 0.0
    p1x, p1y = 2.2, 2.0
    p2y = 4.0
    R = 1.0

    gmsh.model.add("hertzian")
    geo = gmsh.model.geo

    def pt(x, y, size):
        return geo.addPoint(x, y, 0, size)

    p1 = pt(p0x, p0y, mesh_h)
    p2 = pt(p0x, p1y, mesh_h)
    p3 = pt(p1x, p1y, mesh_h)
    p4 = pt(p1x, p0y, mesh_h)
    p5 = pt(p0x, p2y, mesh_h)
    p6 = pt(p1x, p2y, mesh_h)
    p7 = pt(p1x / 2, p1y + R + 0.05, mesh_h)
    p8 = pt(1.1 - math.sqrt(0.0975), 4, mesh_h)
    p9 = pt(1.1 + math.sqrt(0.0975), 4, mesh_h)
    p10 = pt(p1x / 2, p1y + 0.05, mesh_h)
    p11 = pt(p1x / 2, p1y, mesh_h)          # contact node, target_node = (1.1, 2.0)
    p12 = pt(0.85, 4, mesh_h)
    p13 = pt(1.35, 4, mesh_h)
    p14 = pt(p1x / 2 - 0.12, p1y, mesh_h)
    p15 = pt(p1x / 2 + 0.12, p1y, mesh_h)

    l1 = geo.addLine(p1, p2)
    l2 = geo.addLine(p2, p14)
    l15 = geo.addLine(p14, p11)
    l3 = geo.addLine(p3, p4)
    l4 = geo.addLine(p4, p1)
    l5 = geo.addLine(p5, p2)
    l6 = geo.addLine(p11, p15)
    l16 = geo.addLine(p15, p3)
    l7 = geo.addLine(p3, p6)
    l8 = geo.addCircleArc(p8, p7, p10)
    l9 = geo.addCircleArc(p10, p7, p9)
    l10 = geo.addLine(p9, p13)
    l13 = geo.addLine(p13, p12)
    l14 = geo.addLine(p12, p8)
    l11 = geo.addLine(p8, p5)
    l12 = geo.addLine(p6, p9)

    cl1 = geo.addCurveLoop([l1, l2, l15, l6, l16, l3, l4])   # elastic half space
    s1 = geo.addPlaneSurface([cl1])

    cl2 = geo.addCurveLoop([l10, l13, l14, l8, l9])          # cylinder (hole)
    s2 = geo.addPlaneSurface([cl2])

    cl3 = geo.addCurveLoop([l5, l2, l15, l6, l16, l7, l12, l10, l13, l14, l11])
    s3 = geo.addPlaneSurface([cl3, cl2])                     # third medium

    geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [s1], 1, "elastic_half_space")
    gmsh.model.addPhysicalGroup(2, [s2], 2, "contact_body")
    gmsh.model.addPhysicalGroup(2, [s3], 3, "third_medium")
    gmsh.model.addPhysicalGroup(1, [l4], 10, "bottom_boundary")
    gmsh.model.addPhysicalGroup(1, [l13], 11, "top_boundary")

    field = gmsh.model.mesh.field.add("Ball")
    gmsh.model.mesh.field.setNumber(field, "VIn", 0.02 * mesh_h)
    gmsh.model.mesh.field.setNumber(field, "VOut", mesh_h)
    gmsh.model.mesh.field.setNumber(field, "XCenter", 1.1)
    gmsh.model.mesh.field.setNumber(field, "YCenter", 2.025)
    gmsh.model.mesh.field.setNumber(field, "Radius", 0.1)
    gmsh.model.mesh.field.setNumber(field, "Thickness", 0.2)
    gmsh.model.mesh.field.setAsBackgroundMesh(field)

    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)   # Blossom
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)


def generate_mesh(mesh_h, mesh_order, comm, gdim=2, model_rank=0,
                  write_msh_path=None, verbose=False):
    """Quad mesh at size `mesh_h`; returns dolfinx MeshData (mesh, cell_tags, facet_tags).
    mesh_order : geometry order (1 = 4-node quads, 2 = 9-node quads, ...)."""
    from dolfinx.io import gmsh as gmshio

    gmsh.initialize()
    try:
        if not verbose:
            gmsh.option.setNumber("General.Terminal", 0)

        if comm.rank == model_rank:
            _build_geometry(mesh_h)
            gmsh.model.mesh.generate(2)
            if mesh_order > 1:
                gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
                gmsh.model.mesh.setOrder(mesh_order)
            if write_msh_path is not None:
                gmsh.write(write_msh_path)

        mesh_data = gmshio.model_to_mesh(gmsh.model, comm, model_rank, gdim=gdim)
    finally:
        gmsh.finalize()
    return mesh_data


# ============================================================================
# Regularisation
# ============================================================================
def huhu_regularization(u, v, dx, k_r, body):
    Hu, Hv = grad(grad(u)), grad(grad(v))
    return k_r * inner(Hu, Hv) * dx(body)


def huhu_lulu_regularization(u, v, dx, k_h, k_l, body):
    Hu, Hv = grad(grad(u)), grad(grad(v))
    Lu, Lv = div(grad(u)), div(grad(v))
    return (k_h * inner(Hu, Hv) - k_l * inner(Lu, Lv) / len(u)) * dx(body)


def tan_R_regularization(F, p, d, dx, beta_1, beta_2, body=3):
    """Energy term; needs a mixed formulation (extra field p), so it is NOT used
    by the displacement-only main script."""
    psi_tan_R = (beta_1 / 2 * ((F[0, 1] - F[1, 0]) / (F[0, 0] + F[1, 1]) - 1 / d * p) ** 2
                 + beta_2 / 2 * inner(grad(p), grad(p)))
    return psi_tan_R * dx(body)


# ============================================================================
# Material models
# ============================================================================
MATERIAL_MODELS = {}

def register_material(name, make, psi, psi_medium):
    """make(mu, K) -> dict of parameters;  psi(mat, kin) -> energy density;
    psi_medium(mat, kin) -> energy density used for the third medium."""
    MATERIAL_MODELS[name] = dict(make=make, psi=psi, psi_medium=psi_medium)


def make_material(mu, K, model="hartmann_neff"):
    if model not in MATERIAL_MODELS:
        raise ValueError(f"Unknown material model '{model}'. Available: {list(MATERIAL_MODELS)}")
    mat = MATERIAL_MODELS[model]["make"](float(mu), float(K))
    mat.update(model=model, mu=float(mu), K=float(K))
    return mat


def psi(mat, kin):
    return MATERIAL_MODELS[mat["model"]]["psi"](mat, kin)


def psi_2(mat, kin):
    """Energy of the third medium (usually the isochoric part)."""
    return MATERIAL_MODELS[mat["model"]]["psi_medium"](mat, kin)


def young_poisson(mat):
    mu, K = mat["mu"], mat["K"]
    return 9 * mu * K / (3 * K + mu), (3 * K - 2 * mu) / (2 * (3 * K + mu))


# ---- Compressible neo-Hookean ----
def _neo_hooke_make(mu, K):
    return dict()


def _neo_hooke_iso(mat, kin):
    return (mat["mu"] / 2) * (kin.Ib1 - 3)


def _neo_hooke_psi(mat, kin):
    return (mat["K"] / 2) * ln(kin.J) ** 2 + _neo_hooke_iso(mat, kin)


register_material("neo_hooke", _neo_hooke_make, _neo_hooke_psi, _neo_hooke_iso)


# ---- Hartmann-Neff ----
def _hartmann_neff_make(mu, K):
    a, c10, c01 = 0.00367, 0.1788, 0.1958
    mu0_paper = 2.0 * (27.0 * a + c10 + 1.5 * np.sqrt(3.0) * c01)   # ~ 1.573
    s = mu / mu0_paper
    return dict(alpha=s * a, c10=s * c10, c01=s * c01)


def _hartmann_neff_vol(mat, kin):
    return mat["K"] / 50.0 * (kin.J ** 5 + kin.J ** (-5) - 2.0)


def _hartmann_neff_iso(mat, kin):
    return (mat["alpha"] * (kin.Ib1 ** 3 - 27.0)
            + mat["c10"] * (kin.Ib1 - 3.0)
            + mat["c01"] * (kin.Ib2 ** 1.5 - 3.0 * np.sqrt(3.0)))


def _hartmann_neff_psi(mat, kin):
    return _hartmann_neff_vol(mat, kin) + _hartmann_neff_iso(mat, kin)


register_material("hartmann_neff", _hartmann_neff_make, _hartmann_neff_psi, _hartmann_neff_iso)