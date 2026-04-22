import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# AE510 HW8 - Part 1 in Python (single Quad4 element)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

# -------------------------
# Inputs (SI units)
# -------------------------
co = np.array([
    [0.0, 0.0],
    [2.0, 0.0],
    [2.0, 1.0],
    [0.0, 2.0],
], dtype=float)  # node coords: 1-2-3-4 (CCW)

E = 70e9              # Pa
nu = 0.3              # -
t = 1.0               # m
bf = np.array([0.0, 1e6], dtype=float)  # N/m^3, [fx, fy]
tf = np.array([1e6, 0.0], dtype=float)  # N/m^2, traction on side 2-3


# -------------------------
# Core FEM functions
# -------------------------
def D_planestrain(E, nu):
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    M = np.array([
        [1.0 - nu, nu, 0.0],
        [nu, 1.0 - nu, 0.0],
        [0.0, 0.0, 0.5 - nu],
    ])
    return c * M


def gauss2x2():
    g = 1.0 / np.sqrt(3.0)
    gps = np.array([
        [-g, -g],
        [ g, -g],
        [ g,  g],
        [-g,  g],
    ], dtype=float)
    w = np.ones(4)
    return gps, w


def shape_quad4(xi, eta):
    # Reference-space nodes: N1(-1,-1), N2(1,-1), N3(1,1), N4(-1,1)
    xi_i = np.array([-1.0, 1.0, 1.0, -1.0])
    eta_i = np.array([-1.0, -1.0, 1.0, 1.0])

    N = 0.25 * (1.0 + xi_i * xi) * (1.0 + eta_i * eta)
    dN_dxi = 0.25 * xi_i * (1.0 + eta_i * eta)
    dN_deta = 0.25 * eta_i * (1.0 + xi_i * xi)
    return N, dN_dxi, dN_deta


def jacobian_quad4(co, dN_dxi, dN_deta):
    dN_dxi_eta = np.vstack((dN_dxi, dN_deta))  # shape (2,4)
    J = dN_dxi_eta @ co                         # shape (2,2)
    detJ = np.linalg.det(J)
    if detJ <= 0:
        raise ValueError("Non-positive detJ; check node ordering/geometry.")
    dN_dxdy = np.linalg.solve(J, dN_dxi_eta)    # shape (2,4)
    return J, detJ, dN_dxdy


def Nmat_quad4(N):
    Nmat = np.zeros((2, 8))
    Nmat[0, 0::2] = N
    Nmat[1, 1::2] = N
    return Nmat


def Bmat_quad4(dN_dxdy):
    B = np.zeros((3, 8))
    B[0, 0::2] = dN_dxdy[0, :]
    B[1, 1::2] = dN_dxdy[1, :]
    B[2, 0::2] = dN_dxdy[1, :]
    B[2, 1::2] = dN_dxdy[0, :]
    return B


def element_matrices_quad4(co, D, t, bf, tf, gps, w):
    Ke = np.zeros((8, 8))
    fe_body = np.zeros(8)
    fe_trac = np.zeros(8)

    # Area integration (2x2)
    for ig in range(4):
        xi, eta = gps[ig]
        N, dN_dxi, dN_deta = shape_quad4(xi, eta)
        _, detJ, dN_dxdy = jacobian_quad4(co, dN_dxi, dN_deta)

        B = Bmat_quad4(dN_dxdy)
        Nmat = Nmat_quad4(N)

        Ke += (B.T @ D @ B) * detJ * t * w[ig]
        fe_body += (Nmat.T @ bf) * detJ * t * w[ig]

    # Traction integration on side 2-3 (xi = +1, eta in [-1,1])
    for eta in (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)):
        xi = 1.0
        N, _, dN_deta = shape_quad4(xi, eta)
        Nmat = Nmat_quad4(N)

        dX_deta = dN_deta @ co
        Jline = np.linalg.norm(dX_deta)

        fe_trac += (Nmat.T @ tf) * Jline * t

    return Ke, fe_body, fe_trac


def solve_with_fixed_nodes(K, F, fixed_nodes_1based):
    ndof = F.shape[0]
    fixed = []
    for n in fixed_nodes_1based:
        fixed.extend([2 * (n - 1), 2 * (n - 1) + 1])
    fixed = np.array(sorted(fixed), dtype=int)

    all_dofs = np.arange(ndof)
    free = np.setdiff1d(all_dofs, fixed)

    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    R = K @ u - F
    return u, R


def stress_at_point(co, u8, D, xi, eta):
    _, dN_dxi, dN_deta = shape_quad4(xi, eta)
    _, _, dN_dxdy = jacobian_quad4(co, dN_dxi, dN_deta)
    B = Bmat_quad4(dN_dxdy)
    return D @ B @ u8


# -------------------------
# Visualization helpers
# -------------------------
def plot_mesh_lines_ax(ax, xy, conn, line_color='r', lw=0.5):
    """Draw quad mesh edges. line_color: named string, hex, or (R,G,B) tuple."""
    for e in range(conn.shape[0]):
        n = conn[e]
        pts = xy[[n[0], n[1], n[2], n[3], n[0]], :]
        ax.plot(pts[:, 0], pts[:, 1], color=line_color, lw=lw)


def get_stress_value(sig, component):
    """Scalar from sig=[sx,sy,txy].
    component: 'sigma_x' | 'sigma_y' | 'sigma_magnitude' (von Mises)."""
    if component == 'sigma_x':
        return float(sig[0])
    elif component == 'sigma_y':
        return float(sig[1])
    elif component == 'sigma_magnitude':
        sx, sy, txy = float(sig[0]), float(sig[1]), float(sig[2])
        return np.sqrt(sx**2 - sx*sy + sy**2 + 3.0*txy**2)
    else:
        raise ValueError("component must be 'sigma_x', 'sigma_y', or 'sigma_magnitude'")


def nodal_stress_field(xy, conn, u, D, component='sigma_x'):
    """Nodal stress by averaging element corner (xi=±1, eta=±1) evaluations."""
    nnode = xy.shape[0]
    stress_sum = np.zeros(nnode)
    count = np.zeros(nnode, dtype=int)
    corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]

    for e in range(conn.shape[0]):
        nodes = conn[e]
        xy_e = xy[nodes]
        ue = np.zeros(8)
        for a in range(4):
            ue[2*a]     = u[2*nodes[a]]
            ue[2*a + 1] = u[2*nodes[a] + 1]
        for c, (xi, eta) in enumerate(corners):
            sig = stress_at_point(xy_e, ue, D, xi, eta)
            stress_sum[nodes[c]] += get_stress_value(sig, component)
            count[nodes[c]] += 1

    return stress_sum / count


def plot_stress_field_ax(ax, xy, conn, u, D, component='sigma_x',
                          edge_color='r', cmap='jet'):
    """Contourf-style stress via triangulated quads + nodal averaging.
    Returns the tricontourf mappable (pass to plt.colorbar)."""
    import matplotlib.tri as mtri

    sn = nodal_stress_field(xy, conn, u, D, component)

    nelem = conn.shape[0]
    tris = np.zeros((2 * nelem, 3), dtype=int)
    for e in range(nelem):
        n = conn[e]
        tris[2*e]     = [n[0], n[1], n[2]]
        tris[2*e + 1] = [n[0], n[2], n[3]]

    triang = mtri.Triangulation(xy[:, 0], xy[:, 1], tris)
    tcf = ax.tricontourf(triang, sn, levels=20, cmap=cmap)
    ax.triplot(triang, color=edge_color, lw=0.3, alpha=0.6)
    return tcf


# -------------------------
# Main
# -------------------------
def main():
    D = D_planestrain(E, nu)
    gps, w = gauss2x2()

    print("=== Part 1.1: shape/Jacobian table at 2x2 Gauss points ===")
    for i, (xi, eta) in enumerate(gps, start=1):
        N, dN_dxi, dN_deta = shape_quad4(xi, eta)
        J, detJ, dN_dxdy = jacobian_quad4(co, dN_dxi, dN_deta)

        print(f"\nGP {i}: xi={xi:+.6f}, eta={eta:+.6f}")
        print("N      =", np.array2string(N, precision=6, suppress_small=True))
        print("dN/dxi =", np.array2string(dN_dxi, precision=6, suppress_small=True))
        print("dN/deta=", np.array2string(dN_deta, precision=6, suppress_small=True))
        print("dN/dx  =", np.array2string(dN_dxdy[0], precision=6, suppress_small=True))
        print("dN/dy  =", np.array2string(dN_dxdy[1], precision=6, suppress_small=True))
        print(f"detJ   = {detJ:.6f}")
        print("J =\n", J)

    Ke, fe_body, fe_trac = element_matrices_quad4(co, D, t, bf, tf, gps, w)
    F = fe_body + fe_trac

    print("\n=== Part 1.2: body force vector ===")
    print(fe_body)
    print("=== Part 1.3: traction force vector (side 2-3) ===")
    print(fe_trac)
    print("=== Part 1.4: element stiffness matrix Ke ===")
    print(np.array2string(Ke, precision=3, suppress_small=False,
                          floatmode="fixed", max_line_width=220))

    u, R = solve_with_fixed_nodes(Ke, F, fixed_nodes_1based=[1, 4])

    print("=== Nodal displacement [u1 v1 u2 v2 u3 v3 u4 v4]^T ===")
    print(u)
    print("=== Reaction vector ===")
    print(R)

    print("=== Stresses [sigma_x sigma_y tau_xy] at Gauss points ===")
    for i, (xi, eta) in enumerate(gps, start=1):
        sig = stress_at_point(co, u, D, xi, eta)
        print(f"GP {i}: {sig}")

    # ----- Visualization -----
    # Toggle these two variables to change output:
    mesh_line_color  = 'r'         # 'r', 'k', 'b', '#FF0000', (R,G,B), etc.
    stress_component = 'sigma_y'  # 'sigma_x' | 'sigma_y' | 'sigma_magnitude'

    conn_single = np.array([[0, 1, 2, 3]], dtype=int)  # 0-based, single element

    # Undeformed mesh
    fig_um, ax_um = plt.subplots(figsize=(6, 4), constrained_layout=True)
    plot_mesh_lines_ax(ax_um, co, conn_single, line_color=mesh_line_color, lw=1.5)
    ax_um.set_aspect('equal'); ax_um.grid(True, ls=':')
    ax_um.set_xlabel('x [m]'); ax_um.set_ylabel('y [m]')
    ax_um.set_title('Single-Element Mesh (undeformed)')
    fig_um.savefig('part1_quad_undeformed_mesh.png', dpi=180)
    plt.close(fig_um)

    # Deformed mesh
    umax_val  = max(float(np.max(np.abs(u))), 1e-16)
    def_scale = 0.15 * 2.0 / umax_val
    uv        = u.reshape(-1, 2)
    co_def    = co + def_scale * uv

    fig_dm, ax_dm = plt.subplots(figsize=(6, 4), constrained_layout=True)
    plot_mesh_lines_ax(ax_dm, co,     conn_single, line_color='0.6',          lw=1.0)
    plot_mesh_lines_ax(ax_dm, co_def, conn_single, line_color=mesh_line_color, lw=1.5)
    ax_dm.set_aspect('equal'); ax_dm.grid(True, ls=':')
    ax_dm.set_xlabel('x [m]'); ax_dm.set_ylabel('y [m]')
    ax_dm.set_title(f'Deformed Mesh (scale={def_scale:.2e}), gray=undeformed')
    fig_dm.savefig('part1_quad_deformed_mesh.png', dpi=180)
    plt.close(fig_dm)

    # Stress field (contourf-style via triangulated patch + nodal averaging)
    fig_sf, ax_sf = plt.subplots(figsize=(6, 4), constrained_layout=True)
    tcf = plot_stress_field_ax(ax_sf, co, conn_single, u, D,
                                component=stress_component,
                                edge_color=mesh_line_color)
    plt.colorbar(tcf, ax=ax_sf, label='Pa')
    ax_sf.set_aspect('equal')
    ax_sf.set_xlabel('x [m]'); ax_sf.set_ylabel('y [m]')
    ax_sf.set_title(f'Stress: {stress_component}')
    fig_sf.savefig(f'part1_quad_stress_{stress_component}.png', dpi=180)
    plt.close(fig_sf)

    print(f"\nSaved: part1_quad_undeformed_mesh.png, "
          f"part1_quad_deformed_mesh.png, "
          f"part1_quad_stress_{stress_component}.png")


if __name__ == "__main__":
    output_path = "quad_output.txt"
    original_stdout = sys.stdout
    with open(output_path, "w", encoding="utf-8") as f:
        sys.stdout = Tee(original_stdout, f)
        try:
            main()
        finally:
            sys.stdout.flush()
            sys.stdout = original_stdout
