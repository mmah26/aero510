import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.ticker import FormatStrFormatter


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
        ax.text(x, y, z + zoff, f"N{i}", color="k", fontsize=8)

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
        ax.text(x, y, z + zoff, f"N{i}", color="k", fontsize=8)

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

