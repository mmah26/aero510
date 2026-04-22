import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# given parameters
w_drone = 90.0  # drone weight (lb)
E_al = 70e9        # Young's Modulus of Aluminum (Pa)
rho_al = 169.0  # aluminum density (lb/ft^3)

# Member dimensions
w_outer = 4.0 / 12.0    # width of outer hexagonal members (ft)
w_inner = 2.0 / 12.0    # width of inner members (ft)
t = 0.5 / 12.0          # thickness of memberes (ft)

# x, y, z coordinates of hexagonal usv
co = np.array([
    [0, 0, 0],
    [39, 0, 0],
    [28, 32.5, 0],
    [-28, 32.5, 0],
    [-39, 0, 0],
    [-15, -32.5, 0],
    [15, -32.5, 0],
], dtype=float)
co /= 12.0

# element - node connectivity 
e = np.array([
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [0, 5],
    [0, 6],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 1],
], dtype=int)

def hex_geometry(hub_z=0.0, rim_z=0.0):
    """Return node coordinates (ft, 3D) and element connectivity (0-based).

    hub_z_ft: z-coordinate of center node (node 0)
    rim_z_ft: z-coordinate of outer ring nodes (nodes 1..6)
    """
    co[0, 2] = hub_z
    co[1:, 2] = rim_z
    return co, e

def truss_areas(e):
    """Return per-element cross-sectional area vector (ft^2)."""
    outer_el_idx = np.array([6, 7, 8, 9, 10, 11], dtype=int)
    inner_el_idx = np.array([0, 1, 2, 3, 4, 5], dtype=int)
    A = np.zeros(e.shape[0], dtype=float)
    A[inner_el_idx] = w_inner * t
    A[outer_el_idx] = w_outer * t
    return A

def solve_struct_weight(co, e):
    A = truss_areas(e)
    n1s = co[e[:, 0]]   # coordinates of first nodes in e matrix
    n2s = co[e[:, 1]]   # coordinates of second nodes in e matrix
    lengths = np.linalg.norm(n2s - n1s, axis=1)
    V_total = np.sum(lengths * A)
    W_struct = rho_al * V_total
    return {
        "lengths_ft": lengths,
        "V_total_ft3": V_total,
        "W_struct_lb": W_struct,
        "is_valid": W_struct <= w_drone,
    }

def plot_hex_2D(co, e):
    co_2d = co[:, :2]   # take only x, y coordinates
    fig, ax = plt.subplots()
    x = co_2d[e, 0].T
    y = co_2d[e, 1].T
    ax.plot(x, y, "k-", lw=1.5)
    ax.plot(co_2d[:, 0], co_2d[:, 1], "ko", ms=5)

    for i, (xn, yn) in enumerate(co_2d):
        ax.annotate(
            f"N{i}: ({xn:.1f}, {yn:.1f})",
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
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "hex_render.svg", format="svg")
    plt.show()

if __name__ == "__main__":
    co, e = hex_geometry()
    out = solve_struct_weight(co=co, e=e)
    print(f"Total volume (ft^3): {out['V_total_ft3']:.4f}")
    print(f"Structural weight (lb): {out['W_struct_lb']:.2f}")
    print(f"Valid under 90 lb limit: {out['is_valid']}")
    plot_hex_2D(co=co, e=e)
