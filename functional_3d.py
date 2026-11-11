"""
functional_3d.py -- fully 3D versions of the kinematics / stress chain.

Python names cannot start with a digit, so "3D_kinematics" is kinematics_3d etc.

What changes relative to functional.py / methods.py
---------------------------------------------------
* kinematics_3d   : F = I + grad(u) is the genuine 3x3 gradient. The plane-strain
                    embedding (F_zz = 1, no z-coupling) is gone.
* cauchy_stress_3d: sigma = (1/J) dpsi/dF F^T, identical formula, but now dpsi/dF is
                    the full 3x3 tensor (all nine components, incl. z-coupling).
* psi_3d, psi_2_3d: psi() and psi_2() only read kin.J, kin.Ic, kin.Ib1, kin.Ib2 (checked
                    against methods.py: neo_hooke and hartmann_neff), which kinematics_3d
                    provides with the same names, so they are plain aliases.
* StressPostProcessor3D: your StressPostProcessor with the vertical stress component
                    switched from sigma_22 (2D) to sigma_33 (z). Everything else
                    (regions, DG fields, interface handling in value_at) is inherited.
"""
from types import SimpleNamespace

from ufl import Identity, variable, grad, tr, det, diff

from dolfinx.fem import Expression

from functional import StressPostProcessor, get_interpolation_points, von_mises
from methods import psi as _psi, psi_2 as _psi_2


def kinematics_3d(u):
    """Deformation measures for a 3-component displacement field u."""
    if len(u) != 3:
        raise ValueError(f"kinematics_3d needs a 3-component displacement, got {len(u)}")
    I = Identity(3)
    F = variable(I + grad(u))
    C = variable(F.T * F)
    Ic = variable(tr(C))
    J = variable(det(F))
    Cb = variable(J ** (-2.0 / 3.0) * C)
    Ib1 = variable(J ** (-2.0 / 3.0) * Ic)
    Ib2 = variable(0.5 * (Ib1 ** 2 - tr(Cb * Cb)))
    return SimpleNamespace(F=F, J=J, Ic=Ic, Ib1=Ib1, Ib2=Ib2)


def psi_3d(mat, kin):
    """Stored energy of a 3D body (same material law as the 2D psi)."""
    return _psi(mat, kin)


def psi_2_3d(mat, kin):
    """Third-medium energy (same law as the 2D psi_2; scaled by gamma in the main script)."""
    return _psi_2(mat, kin)


def cauchy_stress_3d(psi_expr, kin):
    """Cauchy stress sigma = (1/J) P F^T, P = dpsi/dF (full 3x3)."""
    return (1.0 / kin.J) * diff(psi_expr, kin.F) * kin.F.T


class StressPostProcessor3D(StressPostProcessor):
    """Same fields as the 2D class; 'sigma22_*' become the vertical stress sigma_zz.
    von Mises was already a full 3D formula and is inherited unchanged."""

    def __init__(self, mesh, cell_tags, sigma_el, sigma_ball, sigma_med, element,
                 component=(2, 2)):
        super().__init__(mesh, cell_tags, sigma_el, sigma_ball, sigma_med, element)
        i, j = component
        pts = get_interpolation_points(self.Vs.element)
        sig = {"el": sigma_el, "cylinder": sigma_ball, "medium": sigma_med}
        self.exprs = {k: Expression(e[i, j], pts) for k, e in sig.items()}
        label = f"{i + 1}{j + 1}"
        for k, f in self.f.items():
            f.name = f"sigma{label}_{'ball' if k == 'cylinder' else k}"