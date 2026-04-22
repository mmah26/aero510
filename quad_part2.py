import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


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
        [g, -g],
        [g, g],
        [-g, g],
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


def jacobian_quad4(xy_e, dN_dxi, dN_deta):
    dN_dxi_eta = np.vstack((dN_dxi, dN_deta))
    J = dN_dxi_eta @ xy_e
    detJ = np.linalg.det(J)
    if detJ <= 0:
        raise ValueError("Non-positive detJ; check node ordering/geometry.")
    dN_dxdy = np.linalg.solve(J, dN_dxi_eta)
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


def element_matrices_quad4(xy_e, D, t, bf, tf, gps, w):
    Ke = np.zeros((8, 8))
    fe_body = np.zeros(8)
    fe_trac = np.zeros(8)

    for ig in range(4):
        xi, eta = gps[ig]
        N, dN_dxi, dN_deta = shape_quad4(xi, eta)
        _, detJ, dN_dxdy = jacobian_quad4(xy_e, dN_dxi, dN_deta)
        B = Bmat_quad4(dN_dxdy)
        Nmat = Nmat_quad4(N)

        Ke += (B.T @ D @ B) * detJ * t * w[ig]
        fe_body += (Nmat.T @ bf) * detJ * t * w[ig]

    # side 2-3: xi = +1
    for eta in (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)):
        xi = 1.0
        N, _, dN_deta = shape_quad4(xi, eta)
        Nmat = Nmat_quad4(N)
        dX_deta = dN_deta @ xy_e
        Jline = np.linalg.norm(dX_deta)
        fe_trac += (Nmat.T @ tf) * Jline * t

    return Ke, fe_body, fe_trac


def build_trapezoid_mesh(nx, ny):
    # Domain vertices: (0,0)->(2,0)->(2,1)->(0,2)
    # Mapping from (r,s) in [0,1]^2: x=2r, y=(2-r)s
    xy = np.zeros(((nx + 1) * (ny + 1), 2), dtype=float)

    nid = 0
    for j in range(ny + 1):
        s = j / ny
        for i in range(nx + 1):
            r = i / nx
            x = 2.0 * r
            y = (2.0 - r) * s
            xy[nid, :] = [x, y]
            nid += 1

    conn = np.zeros((nx * ny, 4), dtype=int)
    eid = 0
    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            n1 = (j - 1) * (nx + 1) + i
            n2 = n1 + 1
            n4 = j * (nx + 1) + i
            n3 = n4 + 1
            conn[eid, :] = [n1 - 1, n2 - 1, n3 - 1, n4 - 1]
            eid += 1

    return xy, conn


def assemble_global_sparse(xy, conn, D, t, bf, tf, gps, w):
    nnode = xy.shape[0]
    ndof = 2 * nnode
    K = lil_matrix((ndof, ndof), dtype=float)
    F = np.zeros(ndof)

    for e in range(conn.shape[0]):
        nodes = conn[e, :]
        xy_e = xy[nodes, :]
        Ke, fe_body, fe_trac = element_matrices_quad4(xy_e, D, t, bf, tf, gps, w)

        x2 = xy_e[1, 0]
        x3 = xy_e[2, 0]
        if abs(x2 - 2.0) > 1e-12 or abs(x3 - 2.0) > 1e-12:
            fe_trac[:] = 0.0

        fe = fe_body + fe_trac
        edofs = np.zeros(8, dtype=int)
        for a in range(4):
            edofs[2 * a] = 2 * nodes[a]
            edofs[2 * a + 1] = 2 * nodes[a] + 1

        for i in range(8):
            F[edofs[i]] += fe[i]
            for j in range(8):
                K[edofs[i], edofs[j]] += Ke[i, j]

    return K.tocsr(), F


def solve_system_sparse(K, F, fixed_nodes_zero_based):
    ndof = F.shape[0]
    fixed_dofs = []
    for n in fixed_nodes_zero_based:
        fixed_dofs.extend([2 * n, 2 * n + 1])
    fixed_dofs = np.array(sorted(fixed_dofs), dtype=int)

    all_dofs = np.arange(ndof)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    u = np.zeros(ndof)
    u[free_dofs] = spsolve(K[free_dofs][:, free_dofs], F[free_dofs])
    R = K @ u - F
    return u, R


def element_stress_at_point(xy_e, u_e8, D, xi, eta):
    _, dN_dxi, dN_deta = shape_quad4(xi, eta)
    _, _, dN_dxdy = jacobian_quad4(xy_e, dN_dxi, dN_deta)
    B = Bmat_quad4(dN_dxdy)
    return D @ B @ u_e8


def stress_at_xy_bottom_point(xy, conn, u, D, x_target):
    tol = 1e-12
    for e in range(conn.shape[0]):
        nodes = conn[e, :]
        xy_e = xy[nodes, :]

        if abs(xy_e[0, 1]) < tol and abs(xy_e[1, 1]) < tol:
            x1 = xy_e[0, 0]
            x2 = xy_e[1, 0]
            if x_target >= min(x1, x2) - tol and x_target <= max(x1, x2) + tol:
                xi = 2.0 * (x_target - x1) / (x2 - x1) - 1.0
                eta = -1.0

                ue = np.zeros(8)
                for a in range(4):
                    ue[2 * a] = u[2 * nodes[a]]
                    ue[2 * a + 1] = u[2 * nodes[a] + 1]
                return element_stress_at_point(xy_e, ue, D, xi, eta)

    raise RuntimeError("Could not locate point (x_target,0) in a bottom-edge element.")


def find_tip_node(xy):
    idx = np.where((np.abs(xy[:, 0] - 2.0) < 1e-12) & (np.abs(xy[:, 1] - 1.0) < 1e-12))[0]
    if idx.size == 0:
        raise RuntimeError("Tip node at (2,1) not found.")
    return int(idx[0])


def plot_mesh_lines(ax, xy, conn, color='r', lw=0.4, alpha=1.0):
    """Draw quad mesh edges. color: named string, hex, or (R,G,B) tuple."""
    for e in range(conn.shape[0]):
        n = conn[e]
        pts = xy[[n[0], n[1], n[2], n[3], n[0]], :]
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, alpha=alpha)


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
            sig = element_stress_at_point(xy_e, ue, D, xi, eta)
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
    ax.triplot(triang, color=edge_color, lw=0.25, alpha=0.5)
    return tcf


def make_plots(results, out_dir, D,
               finest_xy=None, finest_conn=None, finest_u=None,
               mesh_line_color='r', stress_component='sigma_y'):
    """
    mesh_line_color : color for mesh lines, e.g. 'r', 'k', 'b', '#FF0000'
    stress_component: 'sigma_x' | 'sigma_y' | 'sigma_magnitude' (von Mises)
    """
    ne  = results[:, 0]
    sx  = results[:, 5]
    sy  = results[:, 6]
    tau = results[:, 7]

    # Instructor-style convergence: absolute stress values vs. number of elements
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(ne, sx,  'b-', lw=1.5, label=r'$\sigma_{xx}$')
    ax.plot(ne, sy,  'r-', lw=1.5, label=r'$\sigma_{yy}$')
    ax.plot(ne, tau, color='goldenrod', lw=1.5, label=r'$\sigma_{xy}$')
    ax.set_xlabel('Number of elements')
    ax.set_ylabel('Stress at (1,0) (Pa)')
    ax.set_title('Convergence of stress at (1,0)')
    ax.legend(); ax.grid(True, ls=':')
    fig.savefig(out_dir / 'quad_part2_convergence_stress.png', dpi=180)
    plt.close(fig)

    if finest_xy is None:
        return

    # Undeformed mesh
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    plot_mesh_lines(ax, finest_xy, finest_conn, color=mesh_line_color, lw=0.5)
    ax.set_aspect('equal'); ax.grid(True, ls=':')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('Undeformed Mesh (finest)')
    fig.savefig(out_dir / 'quad_part2_undeformed_mesh.png', dpi=180)
    plt.close(fig)

    # Deformed mesh
    uv = finest_u.reshape(-1, 2)
    umax = max(float(np.sqrt((uv**2).sum(axis=1)).max()), 1e-16)
    scale = 0.15 / umax
    xy_def = finest_xy + scale * uv

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    plot_mesh_lines(ax, finest_xy, finest_conn, color='0.65', lw=0.35, alpha=0.9)
    plot_mesh_lines(ax, xy_def,    finest_conn, color=mesh_line_color, lw=0.55)
    ax.set_aspect('equal'); ax.grid(True, ls=':')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title(f'Deformed Mesh (scale={scale:.2e}), gray=undeformed')
    fig.savefig(out_dir / 'quad_part2_deformed_mesh.png', dpi=180)
    plt.close(fig)

    # Stress field (contourf-style, triangulated quads, nodal averaging)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    tcf = plot_stress_field_ax(ax, finest_xy, finest_conn, finest_u, D,
                                component=stress_component,
                                edge_color=mesh_line_color)
    plt.colorbar(tcf, ax=ax, label='Pa')
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title(f'Stress: {stress_component} (finest mesh)')
    fig.savefig(out_dir / f'quad_part2_stress_{stress_component}.png', dpi=180)
    plt.close(fig)


def main():
    out_dir = Path('.')

    E = 70e9
    nu = 0.3
    thickness = 1.0
    bf = np.array([0.0, 1e6], dtype=float)
    tf = np.array([1e6, 0.0], dtype=float)

    # Match instructor sweep: nside = 2, 4, ..., 50  →  total = 4, 16, ..., 2500
    mesh_counts = [n * n for n in range(2, 51, 2)]

    D = D_planestrain(E, nu)
    gps, w = gauss2x2()

    print("=== AE510 HW8 Part 2 (Python) ===")
    print("cols: nelem, nside, nnodes, tipUx(2,1), tipUy(2,1), sigma_x(1,0), sigma_y(1,0), tau_xy(1,0), max|u|, ||R_free||_2")

    rows = []
    finest_xy = finest_conn = finest_u = None

    for nelem in mesh_counts:
        nside = int(round(np.sqrt(nelem)))
        if nside * nside != nelem:
            raise ValueError(f"Mesh count {nelem} is not a perfect square.")

        xy, conn = build_trapezoid_mesh(nside, nside)
        K, F = assemble_global_sparse(xy, conn, D, thickness, bf, tf, gps, w)

        fixed_nodes = np.where(np.abs(xy[:, 0]) < 1e-12)[0]
        u, _ = solve_system_sparse(K, F, fixed_nodes)

        # Solver quality check on free dofs
        fixed_dofs = np.sort(np.concatenate((2 * fixed_nodes, 2 * fixed_nodes + 1)))
        all_dofs = np.arange(F.size)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
        r_free = (K @ u - F)[free_dofs]
        res_norm = np.linalg.norm(r_free, ord=2)

        tip_node = find_tip_node(xy)
        tip_ux = u[2 * tip_node]
        tip_uy = u[2 * tip_node + 1]

        sig = stress_at_xy_bottom_point(xy, conn, u, D, x_target=1.0)
        uv = u.reshape(-1, 2)
        umax = np.max(np.sqrt((uv ** 2).sum(axis=1)))

        row = [nelem, nside, xy.shape[0], tip_ux, tip_uy, sig[0], sig[1], sig[2], umax, res_norm]
        rows.append(row)
        print(f"{nelem:6d}  {nside:5d}  {xy.shape[0]:6d}  {tip_ux: .6e}  {tip_uy: .6e}  {sig[0]: .6e}  {sig[1]: .6e}  {sig[2]: .6e}  {umax: .6e}  {res_norm: .3e}")

        finest_xy, finest_conn, finest_u = xy, conn, u

    results = np.array(rows, dtype=float)

    header = (
        "nelem nside nnodes tipUx(m) tipUy(m) "
        "sigma_x_1_0(Pa) sigma_y_1_0(Pa) tau_xy_1_0(Pa) max_u(m) residual_free_norm"
    )
    np.savetxt(out_dir / "quad_part2_results.txt", results, header=header, fmt="%.10e")
    np.savetxt(
        out_dir / "quad_part2_results.csv",
        results,
        delimiter=",",
        header=header.replace(" ", ","),
        comments="",
        fmt="%.10e",
    )

    make_plots(results, out_dir, D,
               finest_xy=finest_xy, finest_conn=finest_conn, finest_u=finest_u)

    print("\nSaved:")
    print("- quad_part2_results.txt")
    print("- quad_part2_results.csv")
    print("- quad_part2_convergence_stress.png   (absolute stress vs n_elements)")
    print("- quad_part2_undeformed_mesh.png")
    print("- quad_part2_deformed_mesh.png")
    print("- quad_part2_stress_<component>.png   (set stress_component in make_plots call)")


if __name__ == "__main__":
    output_path = "quad_part2_output.txt"
    original_stdout = sys.stdout
    with open(output_path, "w", encoding="utf-8") as f:
        sys.stdout = Tee(original_stdout, f)
        try:
            main()
        finally:
            sys.stdout.flush()
            sys.stdout = original_stdout
