"""
stress3d.py -- 3D replacement for StressPostProcessor.

Same public interface as the 2D class used in the main script:
    post = StressPostProcessor3D(mesh, material_tags, sig_e, sig_c, sig_m, stress_element)
    post.fields        -> list of Functions for VTXWriter
    post.vm_total      -> von Mises stress (all three media)
    post.sigma_zz      -> Cauchy sigma_zz (all three media)
    post.update()      -> re-interpolate from the current displacement
    post.value_at((x, y, z), comm=comm) -> sigma_zz at a point (NaN if outside)

sig_* are the 3x3 Cauchy-stress UFL expressions of the substrate, ball and
third medium.  Each is only evaluated in the cells of its own material.
"""
import numpy as np
from ufl import Identity, inner, sqrt, tr
from dolfinx import geometry
from dolfinx.fem import Expression, Function, functionspace


class StressPostProcessor3D:
    def __init__(self, mesh, material_tags, sig_e, sig_c, sig_m, element,
                 tags=(1, 2, 3)):
        self.mesh = mesh
        V0 = functionspace(mesh, ("DG", 0))
        chi = []
        for t in tags:
            f = Function(V0)
            cells = material_tags.find(t)
            f.x.array[:] = 0.0
            f.x.array[V0.dofmap.list[cells].flatten()] = 1.0
            f.x.scatter_forward()
            chi.append(f)
        self._chi = chi

        sig = chi[0] * sig_e + chi[1] * sig_c + chi[2] * sig_m
        dev = sig - tr(sig) / 3.0 * Identity(3)
        vm = sqrt(1.5 * inner(dev, dev))

        self.Vs = functionspace(mesh, element)
        pts = self.Vs.element.interpolation_points
        pts = pts() if callable(pts) else pts      # method in older dolfinx, property in newer
        self.vm_total = Function(self.Vs, name="vm_total")
        self.sigma_zz = Function(self.Vs, name="sigma_zz")
        self._exprs = [(self.vm_total, Expression(vm, pts)),
                       (self.sigma_zz, Expression(sig[2, 2], pts))]
        self.fields = [self.vm_total, self.sigma_zz]
        self._tree = None

    def update(self):
        for f, e in self._exprs:
            f.interpolate(e)
            f.x.scatter_forward()

    def value_at(self, point, comm=None, field=None):
        field = field or self.sigma_zz
        comm = comm or self.mesh.comm
        if self._tree is None:
            self._tree = geometry.bb_tree(self.mesh, self.mesh.topology.dim)
        p = np.zeros((1, 3))
        p[0, :len(point)] = point
        cand = geometry.compute_collisions_points(self._tree, p)
        cells = geometry.compute_colliding_cells(self.mesh, cand, p).links(0)
        val = float(field.eval(p, cells[:1])[0]) if len(cells) > 0 else None
        vals = [v for v in comm.allgather(val) if v is not None]
        return vals[0] if vals else float("nan")
