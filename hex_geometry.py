import numpy as np

# Given parameters in SI units.
LBF_TO_N = 4.4482216152605
IN_TO_M = 0.0254

w_drone = 90.0 * LBF_TO_N           # 90 lbf drone weight ~ 400 N
w_package = 5.0 * LBF_TO_N          # 5 lbf parcel weight ~ 22.25 N
g = 9.80665                         # gravitational acceleration [m/s^2]
E_al = 70e9                         # Young's modulus of aluminum [Pa]
rho_al = 2700.0                     # aluminum density [kg/m^3]

# Member dimensions [m]
w_outer = 4.0 * IN_TO_M             # outer hexagonal member width [m]
w_inner = 2.0 * IN_TO_M             # spoke member width [m]
t = 0.5 * IN_TO_M                   # thickness [m]

# x, y, z coordinates of hexagonal UAV [m]
co = np.array([
    [0, 0, 0],
    [39, 0, 0],
    [28, 32.5, 0],
    [-28, 32.5, 0],
    [-39, 0, 0],
    [-15, -32.5, 0],
    [15, -32.5, 0],
], dtype=float)
co *= IN_TO_M

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
    """Return node coordinates (m, 3D) and element connectivity (0-based).

    hub_z: z-coordinate of center node (node 0) [m]
    rim_z: z-coordinate of outer ring nodes (nodes 1..6) [m]
    """
    co[0, 2] = hub_z
    co[1:, 2] = rim_z
    return co, e

def truss_areas(e):
    """Return per-element cross-sectional area vector (m^2)."""
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
    V_total = np.sum(lengths * A)              # [m^3]
    m_struct = rho_al * V_total               # [kg]
    W_struct = m_struct * g                   # [N]
    return {
        "lengths_m": lengths,
        "V_total_m3": V_total,
        "m_struct_kg": m_struct,
        "W_struct_N": W_struct,
        "is_valid": W_struct <= w_drone,
    }

if __name__ == "__main__":
    co, e = hex_geometry()
    out = solve_struct_weight(co=co, e=e)
    print(f"Total volume (m^3): {out['V_total_m3']:.6f}")
    print(f"Structural mass (kg): {out['m_struct_kg']:.2f}")
    print(f"Structural weight (N): {out['W_struct_N']:.2f}")
    print(f"Valid under drone weight limit ({w_drone:.2f} N): {out['is_valid']}")
