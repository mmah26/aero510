import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


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
    ax.scatter(V[:, 0], V[:, 1], V[:, 2], color="k", s=20)

    for i, (x, y, z) in enumerate(V):
        ax.text(x, y, z, f" N{i}", color="k", fontsize=8)

    for bc in bcs:
        node = bc["node"]
        x, y, z = V[node]
        ax.text(x, y, z, f' {bc["dof"]}=0', color="tab:blue", fontsize=7)

    p_loads = [ld for ld in loads if np.linalg.norm(np.asarray(ld["value"], dtype=float)) > 0]
    q_scale = _load_arrow_scale(V, p_loads)
    for ld in p_loads:
        node = ld["node"]
        fx, fy, fz = np.asarray(ld["value"], dtype=float)
        x, y, z = V[node]
        ax.quiver(x, y, z, fx * q_scale, fy * q_scale, fz * q_scale, color="r", linewidth=1.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    _set_axes_equal_3d(ax, V)
    return ax


def plot_deformed_truss_3d(V, E2N, U_nodes, scale=1.0, ax=None, title="Deformed Truss"):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    V_def = V + scale * U_nodes
    seg0 = np.stack([V[E2N[:, 0]], V[E2N[:, 1]]], axis=1)
    segd = np.stack([V_def[E2N[:, 0]], V_def[E2N[:, 1]]], axis=1)

    lc0 = Line3DCollection(seg0, colors="0.75", linewidths=1.2, linestyles="dashed")
    lcd = Line3DCollection(segd, colors="tab:blue", linewidths=2.0)
    ax.add_collection3d(lc0)
    ax.add_collection3d(lcd)
    ax.scatter(V_def[:, 0], V_def[:, 1], V_def[:, 2], color="tab:blue", s=20)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    _set_axes_equal_3d(ax, np.vstack([V, V_def]))
    return ax


def plot_stress_truss_3d(V, E2N, stress, ax=None, title="Element Axial Stress", cmap="coolwarm"):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    segs = np.stack([V[E2N[:, 0]], V[E2N[:, 1]]], axis=1)
    lc = Line3DCollection(segs, cmap=cmap, linewidths=3.0)
    lc.set_array(np.asarray(stress, dtype=float))
    ax.add_collection3d(lc)
    ax.scatter(V[:, 0], V[:, 1], V[:, 2], color="k", s=14)

    cbar = plt.colorbar(lc, ax=ax, pad=0.12, fraction=0.035)
    cbar.set_label("Axial Stress [Pa]")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    _set_axes_equal_3d(ax, V)
    return ax
