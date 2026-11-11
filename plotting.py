"""Plots for the Hertz sweeps (h-, p- and hp-experiments). Quadrilateral meshes only."""
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

cBlue = np.array([0, 193, 212]) / 255
cRed = np.array([255, 79, 79]) / 255
cYellow = np.array([255, 220, 54]) / 255
cPink = np.array([255, 174, 162]) / 255
plot_colors = [cBlue, cRed, cYellow, cPink]
markers = ["o", "s", "^", "d", "v", "*"]
fig_w_in = (210 / 2) / 25.4
fig_h_in = (297 / 2) / 25.4


def _apply_style():
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12,
        "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "lines.linewidth": 1.5, "lines.markersize": 5,
    })


def _get(r, primary, *fallbacks, default=0.0):
    if primary in r:
        return r[primary]
    for key in fallbacks:
        if key in r:
            return r[key]
    return default


def _grouped_plot(rows, xkey, gkey, ykey, xlabel, ylabel, title, path,
                  logx=True, logy=False, glabel="p", fmt=None):
    """One line per value of `gkey`, y = ykey vs x = xkey. rows: list of dicts."""
    rows = [r for r in rows if r.get(ykey) is not None and r.get(xkey) is not None]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    for i, g in enumerate(sorted({r[gkey] for r in rows})):
        pts = sorted((r[xkey], r[ykey]) for r in rows if r[gkey] == g)
        x, y = zip(*pts)
        lab = f"${glabel}={g}$" if glabel == "p" else f"${glabel}={g:g}$"
        ax.plot(x, y, marker=markers[i % len(markers)],
                color=plot_colors[i % len(plot_colors)], label=lab)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# From the sweep log
# ============================================================================
def from_log(log_path, file_name, dir_name):
    """Parse the sweep table in `log_path` and plot it. Rows follow a line
    'Polynomial degree p = N' and have 6 '|'-separated numbers:
    h | target % | steps | newton | avg newton | time.
    Makes h-plots (one line per p) and, if several p exist, p-plots (one line per h)."""
    records, degree = [], None
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Polynomial degree p\s*=\s*(\d+)", line)
            if m:
                degree = int(m.group(1))
                continue
            parts = [p.strip() for p in line.split("|")]
            if degree is None or len(parts) != 6:
                continue
            try:
                h, disp, steps, newton, avg, t = (float(p) for p in parts)
            except ValueError:
                continue
            records.append(dict(p=degree, h=h, disp_pct=disp, steps=int(steps),
                                newton=int(newton), avg_newton=avg, time=t))
    if not records:
        print(f"No table rows found in {log_path}")
        return
    os.makedirs(dir_name, exist_ok=True)
    _apply_style()

    panels = [
        ("time", "Solver time [s]", "Computational Time", True, "time"),
        ("newton", "Total Newton iterations", "Newton Iterations", False, "newton_total"),
        ("avg_newton", "Avg. Newton / step", "Avg. Newton per Step", False, "newton_avg"),
        ("steps", "Load steps", "Load Steps", False, "steps"),
        ("disp_pct", "Target reached [%]", "Target Reached", False, "target_pct"),
    ]
    multi_p = len({r["p"] for r in records}) > 1
    multi_h = len({r["h"] for r in records}) > 1
    for key, ylabel, title, logy, suffix in panels:
        if multi_h:
            _grouped_plot(records, "h", "p", key, "Mesh size $h$", ylabel, f"{title} vs. Mesh Size",
                          os.path.join(dir_name, f"{file_name}_{suffix}_vs_h.png"), logy=logy)
    print(f"Saved log plots in {dir_name}")


# ============================================================================
# Hertz comparison
# ============================================================================
def hertz_plot(sweep_results, case_name, case_dir, sigma_ana=-822.0):
    """Energy and relative sigma22 error vs h, p and #dofs, plus a (k_lulu, k_huhu)
    heatmap if the regularisation parameters were swept. Written to `case_dir`.
    sigma_ana: analytical Hertz sigma22 at the contact node."""
    _apply_style()
    os.makedirs(case_dir, exist_ok=True)

    rows = []
    for r in sweep_results:
        s = r.get("sigma22_at_target_node")
        rows.append(dict(p=r["fe_degree"], h=r["mesh_h"], dofs=r["num_dofs"],
                         energy=r["energy_bodies"],
                         err=(abs((s - sigma_ana) / sigma_ana) * 100.0 if s is not None else None),
                         kl=r["k_lulu_value"], kh=r["k_huhu_value"]))
    multi_p = len({r["p"] for r in rows}) > 1
    multi_h = len({r["h"] for r in rows}) > 1

    if multi_h:
        _grouped_plot(rows, "h", "p", "energy", "Mesh size $h$", "Bodies Potential Energy",
                      "Bodies Potential Energy vs. Mesh size",
                      os.path.join(case_dir, "bodies_potential_energy_vs_h.png"))
        _grouped_plot(rows, "h", "p", "err", r"Mesh size $h$", r"Relative Stress Error $e_\sigma$ (%)",
                      r"Relative Error in $\sigma_{yy}$ vs. $h$",
                      os.path.join(case_dir, "stress_relative_error_vs_h.png"), logy=True)
    _grouped_plot(rows, "dofs", "p", "err", "Degrees of freedom",
                  r"Relative Stress Error $e_\sigma$ (%)", r"Error vs. DOFs",
                  os.path.join(case_dir, "stress_relative_error_vs_dofs.png"), logy=True)

    # ---- (k_lulu, k_huhu) heatmap ----
    valid = [r for r in rows if r["err"] is not None]
    kl_u = np.unique([r["kl"] for r in valid])
    kh_u = np.unique([r["kh"] for r in valid])
    if len(kl_u) > 1 and len(kh_u) > 1:
        grid = np.full((len(kh_u), len(kl_u)), np.nan)
        for i, hu in enumerate(kh_u):
            for j, lu in enumerate(kl_u):
                m = [r["err"] for r in valid if r["kh"] == hu and r["kl"] == lu]
                if m:
                    grid[i, j] = m[0]
        plt.figure(figsize=(8, 6))
        im = plt.imshow(grid, origin="lower", aspect="auto", cmap="turbo",
                        extent=[kl_u.min(), kl_u.max(), kh_u.min(), kh_u.max()])
        plt.colorbar(im, label="Relative Stress Error (%)")
        plt.xlabel(r"$k_{lulu}$")
        plt.ylabel(r"$k_{huhu}$")
        plt.title(r"Relative Stress Error over $k_{lulu}$ and $k_{huhu}$")
        plt.xticks(kl_u)
        plt.yticks(kh_u)
        plt.tight_layout()
        plt.savefig(os.path.join(case_dir, "error_heatmap_k_lulu_k_huhu.png"), dpi=200)
        plt.close()
    print(f"Saved convergence plots in folder: '{case_dir}'")


# ============================================================================
# Quad-mesh drawing (no triangulation)
# ============================================================================
def _quad_data(mesh, u):
    """Deformed quad corners per cell. Returns (verts (n,4,2), |u| per cell,
    corner node ids (n,4) in drawing order)."""
    from dolfinx import fem
    gdim = mesh.geometry.dim
    V1 = fem.functionspace(mesh, ("Lagrange", 1))
    V1v = fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
    u1 = fem.Function(V1v)
    u1.interpolate(u)
    xy = V1.tabulate_dof_coordinates()[:, :2]
    uv = u1.x.array.reshape(-1, gdim)
    xyd = xy + uv
    n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    ring = V1.dofmap.list[:n_cells][:, [0, 1, 3, 2]]      # tensor ordering -> polygon ordering
    umag = np.linalg.norm(uv, axis=1)[ring].mean(axis=1)
    return xyd[ring], umag, ring, xyd

def _outline_edges(ring, labels=None):
    """Node pairs on the exterior boundary, plus edges between cells with different labels."""
    edge_to_cells = defaultdict(list)
    for c, quad in enumerate(ring):
        for k in range(4):
            a, b = quad[k], quad[(k + 1) % 4]
            edge_to_cells[(min(a, b), max(a, b))].append(c)
    out = []
    for key, cs in edge_to_cells.items():
        if len(cs) == 1 or (labels is not None and len(cs) == 2 and labels[cs[0]] != labels[cs[1]]):
            out.append(key)
    return out

def plot_fields(mesh, u, vm_dg0, path, material_tags=None, title_suffix=""):
    """Two panels on the deformed mesh (turbo colour map, one colour per cell):
    |u| (left) and von Mises stress (right). The outlines of the material
    domains (cell tags) are drawn on both panels. Serial only."""
    from functional import cell_mean
 
    verts, umag, ring, xyd = _quad_data(mesh, u)
    n = len(verts)
    vm = cell_mean(vm_dg0)[:n]
 
    # material-tag outline: exterior boundary + edges between cells with different tags
    labels = None
    if material_tags is not None:
        imap = mesh.topology.index_map(mesh.topology.dim)
        lab = np.zeros(imap.size_local + imap.num_ghosts, dtype=np.int32)
        lab[material_tags.indices] = material_tags.values
        labels = lab[:n]
    segs = [[tuple(xyd[a]), tuple(xyd[b])] for a, b in _outline_edges(ring, labels)]
 
    xs, ys = verts[..., 0], verts[..., 1]
    panel_w = 7.0
    panel_h = max(panel_w * np.ptp(ys) / max(np.ptp(xs), 1e-12), 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(2 * panel_w + 2, panel_h + 1.0),
                             layout="constrained")
 
    for ax, vals, title, cbar_label in (
            (axes[0], umag, "Deformation - |u|", "|u|"),
            (axes[1], vm, "von Mises stress", r"$\sigma_{vM}$")):
        pc = PolyCollection(verts, array=vals, cmap="turbo", edgecolors="face", linewidths=0.2)
        ax.add_collection(pc)
        ax.add_collection(LineCollection(segs, colors="k", linewidths=0.8))
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(ys.min(), ys.max())
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("X position")
        fig.colorbar(pc, ax=ax, location="top", pad=0.08, shrink=0.85, label=cbar_label)
    axes[0].set_ylabel("Y position")
    if title_suffix:
        fig.suptitle(title_suffix, fontsize=11)
 
    fig.savefig(path, dpi=200)
    plt.close(fig)
# ============================================================================
# Summary plots from the list returned by run_case()
# ============================================================================
def create_plots(file_name, dir_name, sweep_results, results=None):
    """Energy, time and deformed material-domain snapshots."""
    if not sweep_results and not results:
        return
    if results is None:
        results = sweep_results
    ref = results if results else sweep_results
    comm = ref[0]["mesh"].comm
    if comm.rank != 0:
        return
    os.makedirs(dir_name, exist_ok=True)
    _apply_style()

    def rows_of(rs):
        return [dict(p=r["fe_degree"], h=_get(r, "mesh_h", "h"),
                     energy=_get(r, "potential", "energy"),
                     time=_get(r, "time_total", "solver_time")) for r in rs]

    res_rows, sw_rows = rows_of(results), rows_of(sweep_results)
    if len({r["h"] for r in res_rows}) > 1:
        _grouped_plot(res_rows, "h", "p", "energy", "Mesh size $h$", "Total Potential Energy",
                      "Convergence of Potential Energy vs. Mesh Size",
                      os.path.join(dir_name, f"{file_name}_potential_energy_vs_h.png"))
        _grouped_plot([r for r in sw_rows if r["time"] > 0], "h", "p", "time", "Mesh size $h$",
                      "Computational Time [s]", "Computational Time vs. Mesh Size",
                      os.path.join(dir_name, f"{file_name}_time_vs_h.png"), logy=True)

    # ---- deformed material domains ----
    if comm.size > 1:
        print("Deformed-domain plots skipped (serial only).")
        return
    for case in (r for r in results if "u" in r and "mesh" in r):
        mesh0, u = case["mesh"], case["u"]
        deg = _get(case, "fe_degree", default=0)
        h_val = _get(case, "mesh_h", "h")
        verts, _, ring, xyd = _quad_data(mesh0, u)

        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
        tags = case.get("material_tags")
        labels = None
        if tags is not None:
            n_all = mesh0.topology.index_map(mesh0.topology.dim).size_local
            lab = np.zeros(n_all, dtype=np.int32)
            lab[tags.indices] = tags.values
            labels = lab[:len(verts)]
            uniq = sorted(np.unique(labels).tolist())
            idx_of = {m: i for i, m in enumerate(uniq)}
            col = np.array([idx_of[m] for m in labels])
            pc = PolyCollection(verts, array=col, cmap=ListedColormap(plot_colors[:len(uniq)]),
                                edgecolors="face", linewidths=0.2)
            pc.set_clim(-0.5, len(uniq) - 0.5)
            ax.legend(handles=[Patch(facecolor=plot_colors[idx_of[m]], edgecolor="k",
                                     label=f"Material {m}") for m in uniq],
                      loc="upper right", fontsize=9, framealpha=0.9)
        else:
            pc = PolyCollection(verts, facecolors=[plot_colors[0]], edgecolors="face")
        ax.add_collection(pc)
        segs = [[tuple(xyd[a]), tuple(xyd[b])] for a, b in _outline_edges(ring, labels)]
        ax.add_collection(LineCollection(segs, colors="k", linewidths=0.8))
        ax.set_xlim(verts[..., 0].min(), verts[..., 0].max())
        ax.set_ylim(verts[..., 1].min(), verts[..., 1].max())
        ax.set(xlabel="X position", ylabel="Y position",
               title=f"Deformed Material Domains ($p={deg}$, $h={h_val:g}$)")
        ax.set_aspect("equal")
        fig.tight_layout()
        out = os.path.join(dir_name, f"{file_name}_p{deg}_h{h_val:.2e}_final_deformation.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved material-domain plot for p={deg}, h={h_val:.2e} to: {out}")