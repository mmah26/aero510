import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D


def _set_axes_equal_3d(ax, pts):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = np.max(maxs - mins)
    if span <= 0:
        span = 1.0
    half = 0.5 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def _style_3d_axes_clean(ax):
    ax.grid(False)
    # keep pane background, but disable gridlines
    try:
        ax.xaxis.pane.set_visible(True)
        ax.yaxis.pane.set_visible(True)
        ax.zaxis.pane.set_visible(True)
    except Exception:
        pass
    ax.tick_params(axis="both", which="major", labelsize=8, pad=1)
    ax.tick_params(axis="z", which="major", labelsize=8, pad=1)


def _load_arrow_scale(V, loads):
    if len(loads) == 0:
        return 1.0
    span = np.max(np.ptp(V, axis=0))
    span = max(float(span), 1.0)
    mags = [np.linalg.norm(np.asarray(ld["value"], dtype=float)) for ld in loads]
    max_mag = max(float(np.max(mags)), 1.0)
    return 0.18 * span / max_mag


def _bc_label_from_dofs(dof_set):
    # dof_set entries are {'ux','uy','uz'} etc.
    order = [("ux", "x"), ("uy", "y"), ("uz", "z")]
    comps = [sym for key, sym in order if key in dof_set]
    if len(comps) == 0:
        return None
    if len(comps) == 1:
        return rf"$u_{{{comps[0]}}}=0$"
    return rf"$u_{{{','.join(comps)}}}=0$"


def summarize_case(name, result):
    max_u = result["max_u"]
    stress = result["stress"]
    max_abs_stress = np.max(np.abs(stress))
    print(f"\n{name}")
    print(f"  max nodal deflection magnitude: {max_u:.6e} ft")
    print(f"  element stress range: [{np.min(stress):.6e}, {np.max(stress):.6e}] Pa")
    print(f"  max |element stress|: {max_abs_stress:.6e} Pa")


def deformation_scale(V, U_nodes):
    max_u = np.max(np.linalg.norm(U_nodes, axis=1))
    if max_u <= 0:
        return 1.0
    char_len = max(np.ptp(V[:, 0]), np.ptp(V[:, 1]), np.ptp(V[:, 2]))
    char_len = max(float(char_len), 1.0)
    return 0.15 * char_len / max_u


def write_case_outputs(out_dir, case_tag, E2N, out):
    n_nodes = out["U_nodes"].shape[0]
    node_ids = np.arange(n_nodes, dtype=int)
    node_table = np.column_stack([node_ids, out["U_nodes"], out["u_mag"]])
    node_header = "node,ux_ft,uy_ft,uz_ft,u_mag_ft"
    np.savetxt(
        out_dir / f"{case_tag}_node_displacements.csv",
        node_table,
        delimiter=",",
        header=node_header,
        comments="",
        fmt=["%d", "%.8e", "%.8e", "%.8e", "%.8e"],
    )

    elem_ids = np.arange(E2N.shape[0], dtype=int)
    elem_table = np.column_stack([elem_ids, E2N[:, 0], E2N[:, 1], out["strain"], out["stress"]])
    elem_header = "element,node_i,node_j,strain,stress_pa"
    np.savetxt(
        out_dir / f"{case_tag}_element_results.csv",
        elem_table,
        delimiter=",",
        header=elem_header,
        comments="",
        fmt=["%d", "%d", "%d", "%.8e", "%.8e"],
    )

    with open(out_dir / f"{case_tag}_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"max_nodal_deflection_ft={out['max_u']:.8e}\n")
        f.write(f"min_element_stress_pa={np.min(out['stress']):.8e}\n")
        f.write(f"max_element_stress_pa={np.max(out['stress']):.8e}\n")
        f.write(f"max_abs_element_stress_pa={np.max(np.abs(out['stress'])):.8e}\n")


def plot_hex_2D(co, e, out_path=None, show=False):
    co_2d = co[:, :2]
    fig, ax = plt.subplots()
    x = co_2d[e, 0].T
    y = co_2d[e, 1].T
    ax.plot(x, y, "k-", lw=1.5)
    ax.plot(co_2d[:, 0], co_2d[:, 1], "ko", ms=5)

    for i, (xn, yn) in enumerate(co_2d):
        zn = co[i, 2] if co.shape[1] > 2 else 0.0
        ax.annotate(
            f"N{i}: ({xn:.1f}, {yn:.1f}, {zn:.1f})",
            xy=(xn, yn),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="b",
        )

    mid = co_2d[e].mean(axis=1)
    for k, (xm, ym) in enumerate(mid):
        ax.annotate(
            f"E{k}",
            xy=(xm, ym),
            xytext=(4, 4),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8,
            color="r",
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (ft)")
    ax.set_ylabel("y (ft)")
    ax.set_title("2D Hex Truss")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if out_path is not None:
        fig.savefig(out_path, format="svg")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_truss_with_bcs_loads_3d(V, E2N, bcs, loads, ax=None, title="Truss with BCs and Loads"):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    segs = np.stack([V[E2N[:, 0]], V[E2N[:, 1]]], axis=1)
    lc = Line3DCollection(segs, colors="0.25", linewidths=1.6)
    ax.add_collection3d(lc)
    ax.scatter(V[:, 0], V[:, 1], V[:, 2], color="darkgreen", edgecolors="darkgreen", s=26, alpha=1.0)

    # Node labels (slightly offset in +z for readability)
    zoff = max(0.02 * np.max(np.ptp(V, axis=0)), 1e-3)
    for i, (x, y, z) in enumerate(V):
        ax.text(x, y, z + zoff, f"N{i}", color="k", fontsize=8, zorder=20)

    # Element labels at midpoints, slight +z offset.
    mids = segs.mean(axis=1)
    for k, (xm, ym, zm) in enumerate(mids):
        ax.text(xm, ym, zm + 0.45 * zoff, f"E{k}", color="m", fontsize=8, zorder=20)

    # BC labels from case setup dict, offset in -z to avoid node-label overlap.
    bc_by_node = {}
    for bc in bcs:
        if bc.get("type", "dirichlet") != "dirichlet":
            continue
        node = int(bc["node"])
        dof = bc.get("dof")
        if dof is None:
            continue
        bc_by_node.setdefault(node, set()).add(dof)
    for node, dof_set in bc_by_node.items():
        label = _bc_label_from_dofs(dof_set)
        if label is None:
            continue
        x, y, z = V[node]
        ax.text(x, y, z - 3.8 * zoff, label, color="b", fontsize=8, zorder=20)

    p_loads = [ld for ld in loads if np.linalg.norm(np.asarray(ld["value"], dtype=float)) > 0]
    q_scale = _load_arrow_scale(V, p_loads)
    for ld in p_loads:
        node = ld["node"]
        fx, fy, fz = np.asarray(ld["value"], dtype=float)
        x, y, z = V[node]
        dx, dy, dz = fx * q_scale, fy * q_scale, fz * q_scale
        ax.quiver(x, y, z, dx, dy, dz, color="r", linewidth=1.5)
        fmag = np.linalg.norm([fx, fy, fz])
        ax.text(x + dx, y + dy, z + dz, f"{fmag:.2f} lbf", color="r", fontsize=8)

    ax.set_xlabel("x (ft)")
    ax.set_ylabel("y (ft)")
    ax.set_zlabel("z (ft)")
    ax.xaxis.labelpad = 4
    ax.yaxis.labelpad = 4
    ax.zaxis.labelpad = 0
    ax.set_title(title)
    _set_axes_equal_3d(ax, V)
    _style_3d_axes_clean(ax)
    return ax


def plot_deformed_stress_truss_3d(
    V,
    E2N,
    U_nodes,
    stress,
    scale=1.0,
    include_undeformed=True,
    ax=None,
    title="Deformed Truss with Stress",
    cmap="turbo",
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    V_def = V + scale * U_nodes
    seg0 = np.stack([V[E2N[:, 0]], V[E2N[:, 1]]], axis=1)
    segd = np.stack([V_def[E2N[:, 0]], V_def[E2N[:, 1]]], axis=1)

    if include_undeformed:
        lc0 = Line3DCollection(seg0, colors="0.75", linewidths=1.2, linestyles="dashed")
        lc0.set_zorder(1)
        lc0.set_sort_zpos(-1e9)
        ax.add_collection3d(lc0)

    s = np.asarray(stress, dtype=float)
    lcd = Line3DCollection(segd, cmap=cmap, linewidths=3.0)
    lcd.set_array(s)
    lcd.set_clim(np.min(s), np.max(s))
    lcd.set_zorder(3)
    lcd.set_sort_zpos(1e9)
    ax.add_collection3d(lcd)
    ax.scatter(V_def[:, 0], V_def[:, 1], V_def[:, 2], color="darkgreen", edgecolors="darkgreen", s=26, alpha=1.0, zorder=4)

    # Node labels at displaced node locations (offset in +z for readability)
    zoff = max(0.02 * np.max(np.ptp(V_def, axis=0)), 1e-3)
    for i, (x, y, z) in enumerate(V_def):
        ax.text(x, y, z + zoff, f"N{i}", color="k", fontsize=8, zorder=20)

    # Element labels at displaced midpoints, slight +z offset.
    mids = segd.mean(axis=1)
    for k, (xm, ym, zm) in enumerate(mids):
        ax.text(xm, ym, zm + 0.45 * zoff, f"E{k}", color="m", fontsize=8, zorder=20)

    # Max stress info in legend (upper-left), instead of midpoint annotation.
    idx_max = int(np.argmax(np.abs(s)))
    sigma_at_max_mag = float(s[idx_max])
    legend_text = rf"$\sigma_{{max}}={sigma_at_max_mag:.3e}\ \mathrm{{Pa}}$ at E{idx_max}"
    legend_handle = Line2D([], [], linestyle="None", label=legend_text)
    ax.legend(handles=[legend_handle], loc="upper left", fontsize=8, frameon=True)

    cbar = plt.colorbar(lcd, ax=ax, pad=0.12, fraction=0.04, shrink=0.82, aspect=24)
    cbar.set_label("Axial Stress [Pa]")
    cbar.ax.tick_params(labelsize=8)
    cbar.formatter = FormatStrFormatter("%.2e")
    cbar.update_ticks()

    ax.set_xlabel("x (ft)")
    ax.set_ylabel("y (ft)")
    ax.set_zlabel("z (ft)")
    ax.xaxis.labelpad = 4
    ax.yaxis.labelpad = 4
    ax.zaxis.labelpad = 0
    ax.set_title(title)
    _set_axes_equal_3d(ax, np.vstack([V, V_def]))
    _style_3d_axes_clean(ax)
    return ax
