import numpy as np
from dataclasses import dataclass

DOF_MAP = {"ux": 0, "uy": 1}


def dirichlet_bcs_for_edge(xy, edge, dofs=("ux", "uy"), value=0.0, tol=1e-10):
    """
    Build Dirichlet BC dictionaries on a named boundary edge:
    edge in {"left","right","bottom","top"}.
    """
    x = xy[:, 0]
    y = xy[:, 1]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    if edge == "left":
        nodes = np.where(np.abs(x - xmin) <= tol)[0]
    elif edge == "right":
        nodes = np.where(np.abs(x - xmax) <= tol)[0]
    elif edge == "bottom":
        nodes = np.where(np.abs(y - ymin) <= tol)[0]
    elif edge == "top":
        nodes = np.where(np.abs(y - ymax) <= tol)[0]
    else:
        raise ValueError(f'Unsupported edge "{edge}".')

    bcs = []
    for n in nodes:
        for d in dofs:
            bcs.append({"type": "dirichlet", "node": int(n), "dof": d, "value": float(value)})
    return bcs


def traction_load(edge, value):
    """Create a traction load dictionary for a boundary edge."""
    return {"type": "traction_edge", "edge": edge, "value": list(np.asarray(value, dtype=float))}


def D_matrix(E, nu, plane="stress"):
    if plane == "stress":
        c = E / (1.0 - nu**2)
        M = np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - nu)],
            ]
        )
        return c * M
    if plane == "strain":
        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        M = np.array(
            [
                [1.0 - nu, nu, 0.0],
                [nu, 1.0 - nu, 0.0],
                [0.0, 0.0, 0.5 - nu],
            ]
        )
        return c * M
    raise ValueError("plane must be 'stress' or 'strain'.")


def orthotropic_Q_plane_stress(E1, E2, G12, nu12):
    """
    Reduced in-plane stiffness matrix Q (material axes 1-2) for plane stress.
    Uses engineering shear gamma_12 convention.
    """
    nu21 = nu12 * E2 / E1
    den = 1.0 - nu12 * nu21
    if den <= 0.0:
        raise ValueError("Invalid orthotropic properties: 1 - nu12*nu21 must be > 0.")
    Q11 = E1 / den
    Q22 = E2 / den
    Q12 = nu12 * E2 / den
    Q66 = G12
    return np.array(
        [
            [Q11, Q12, 0.0],
            [Q12, Q22, 0.0],
            [0.0, 0.0, Q66],
        ],
        dtype=float,
    )


def rotate_Qbar_plane_stress(Q, theta_deg):
    """
    Transform reduced stiffness Q from material axes (1-2) to global axes (x-y).
    theta_deg: fiber angle measured from +x toward +y.
    """
    t = np.deg2rad(theta_deg)
    m = np.cos(t)
    n = np.sin(t)

    Q11 = Q[0, 0]
    Q22 = Q[1, 1]
    Q12 = Q[0, 1]
    Q66 = Q[2, 2]

    m2 = m * m
    n2 = n * n
    m4 = m2 * m2
    n4 = n2 * n2

    Qb11 = Q11 * m4 + 2.0 * (Q12 + 2.0 * Q66) * m2 * n2 + Q22 * n4
    Qb22 = Q11 * n4 + 2.0 * (Q12 + 2.0 * Q66) * m2 * n2 + Q22 * m4
    Qb12 = (Q11 + Q22 - 4.0 * Q66) * m2 * n2 + Q12 * (m4 + n4)
    Qb16 = (Q11 - Q12 - 2.0 * Q66) * m * m2 * n - (Q22 - Q12 - 2.0 * Q66) * m * n2 * n
    Qb26 = (Q11 - Q12 - 2.0 * Q66) * m * n2 * n - (Q22 - Q12 - 2.0 * Q66) * m * m2 * n
    Qb66 = (Q11 + Q22 - 2.0 * Q12 - 2.0 * Q66) * m2 * n2 + Q66 * (m4 + n4)

    return np.array(
        [
            [Qb11, Qb12, Qb16],
            [Qb12, Qb22, Qb26],
            [Qb16, Qb26, Qb66],
        ],
        dtype=float,
    )


def gauss2x2():
    g = 1.0 / np.sqrt(3.0)
    gps = np.array([[-g, -g], [g, -g], [g, g], [-g, g]], dtype=float)
    w = np.ones(4, dtype=float)
    return gps, w


def shape_quad4(xi, eta):
    xi_i = np.array([-1.0, 1.0, 1.0, -1.0], dtype=float)
    eta_i = np.array([-1.0, -1.0, 1.0, 1.0], dtype=float)
    N = 0.25 * (1.0 + xi_i * xi) * (1.0 + eta_i * eta)
    dN_dxi = 0.25 * xi_i * (1.0 + eta_i * eta)
    dN_deta = 0.25 * eta_i * (1.0 + xi_i * xi)
    return N, dN_dxi, dN_deta


def jacobian_quad4(xy_e, dN_dxi, dN_deta):
    dN_dxi_eta = np.vstack((dN_dxi, dN_deta))  # (2,4)
    J = dN_dxi_eta @ xy_e  # (2,2)
    detJ = np.linalg.det(J)
    if detJ <= 0.0:
        raise ValueError("Non-positive detJ; check element orientation.")
    dN_dxdy = np.linalg.solve(J, dN_dxi_eta)  # (2,4)
    return J, detJ, dN_dxdy


def Nmat_quad4(N):
    Nmat = np.zeros((2, 8), dtype=float)
    Nmat[0, 0::2] = N
    Nmat[1, 1::2] = N
    return Nmat


def Bmat_quad4(dN_dxdy):
    B = np.zeros((3, 8), dtype=float)
    B[0, 0::2] = dN_dxdy[0, :]
    B[1, 1::2] = dN_dxdy[1, :]
    B[2, 0::2] = dN_dxdy[1, :]
    B[2, 1::2] = dN_dxdy[0, :]
    return B


@dataclass
class DirectFEMQuad:
    case: dict

    def __post_init__(self):
        self.xy = np.asarray(self.case["mesh"]["xy"], dtype=float)
        self.conn = np.asarray(self.case["mesh"]["conn"], dtype=int)
        if self.conn.min() == 1:
            self.conn = self.conn - 1

        self.Nn = self.xy.shape[0]
        self.Ne = self.conn.shape[0]
        self.ndim = 2
        self.nne = 4
        self.ndof = self.Nn * self.ndim
        self.dofs = np.arange(self.ndof, dtype=int).reshape(self.Nn, self.ndim)

        props = self.case["properties"]
        self.t = float(props["t"])
        self.plane = props.get("plane", "stress")
        self.theta_deg = float(props.get("theta_deg", 0.0))

        if "D" in props:
            self.D = np.asarray(props["D"], dtype=float)
            if self.D.shape != (3, 3):
                raise ValueError("properties['D'] must be a 3x3 matrix.")
            self.material_model = "custom_D"
        elif all(k in props for k in ("E1", "E2", "G12", "nu12")):
            if self.plane != "stress":
                raise ValueError("Orthotropic input currently supports only plane='stress'.")
            self.E1 = float(props["E1"])
            self.E2 = float(props["E2"])
            self.G12 = float(props["G12"])
            self.nu12 = float(props["nu12"])
            self.Q = orthotropic_Q_plane_stress(self.E1, self.E2, self.G12, self.nu12)
            self.D = rotate_Qbar_plane_stress(self.Q, self.theta_deg)
            self.material_model = "orthotropic"
        else:
            self.E = float(props["E"])
            self.nu = float(props["nu"])
            self.D = D_matrix(self.E, self.nu, self.plane)
            self.material_model = "isotropic"

        self.body_force = np.asarray(self.case.get("body_force", [0.0, 0.0]), dtype=float)
        if self.body_force.size != 2:
            raise ValueError("body_force must have length 2.")

        self.loads = self.case.get("loads", [])
        self.bcs = self.case.get("bcs", [])

        self.xmin = float(np.min(self.xy[:, 0]))
        self.xmax = float(np.max(self.xy[:, 0]))
        self.ymin = float(np.min(self.xy[:, 1]))
        self.ymax = float(np.max(self.xy[:, 1]))
        self.tol = max(np.max(np.ptp(self.xy, axis=0)) * 1e-10, 1e-12)

        self.Kglob = np.zeros((self.ndof, self.ndof), dtype=float)
        self.Fglob = np.zeros(self.ndof, dtype=float)

        self.constrained_dofs, self.constrained_vals = self._parse_dirichlet_bcs()

        self.assembly()
        self.solve()
        self.postprocess()

    def _parse_dirichlet_bcs(self):
        c_dofs = []
        c_vals = []
        for bc in self.bcs:
            if bc.get("type", "dirichlet") != "dirichlet":
                continue
            node = int(bc["node"])
            dof_key = bc["dof"]
            if dof_key not in DOF_MAP:
                raise ValueError(f'Unsupported BC dof "{dof_key}".')
            gdof = self.dofs[node, DOF_MAP[dof_key]]
            c_dofs.append(gdof)
            c_vals.append(float(bc.get("value", 0.0)))
        if len(c_dofs) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        c_dofs = np.asarray(c_dofs, dtype=int)
        c_vals = np.asarray(c_vals, dtype=float)
        uniq, idx = np.unique(c_dofs, return_index=True)
        return uniq, c_vals[idx]

    def _traction_on_element_edge(self, xy_e, edge, tvec):
        # local edge map: (i,j, fixed_coord, fixed_val, varying)
        # edge names are in global panel sense.
        # local sides:
        # 0: n1-n2 (eta=-1), 1: n2-n3 (xi=+1), 2: n3-n4 (eta=+1), 3: n4-n1 (xi=-1)
        fe = np.zeros(8, dtype=float)
        gp = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))

        def edge_line_integral(xi_eta):
            nonlocal fe
            xi, eta = xi_eta
            N, dN_dxi, dN_deta = shape_quad4(xi, eta)
            Nmat = Nmat_quad4(N)
            if np.isclose(abs(xi), 1.0):
                dX_dq = dN_deta @ xy_e
            else:
                dX_dq = dN_dxi @ xy_e
            Jline = np.linalg.norm(dX_dq)
            # tvec is line traction [N/m], integrated along edge length [m].
            fe += (Nmat.T @ tvec) * Jline

        if edge == "right":
            for eta in gp:
                edge_line_integral((1.0, eta))
        elif edge == "left":
            for eta in gp:
                edge_line_integral((-1.0, eta))
        elif edge == "top":
            for xi in gp:
                edge_line_integral((xi, 1.0))
        elif edge == "bottom":
            for xi in gp:
                edge_line_integral((xi, -1.0))
        return fe

    def _element_on_boundary_edge(self, xy_e, edge):
        if edge == "left":
            return np.all(np.abs(xy_e[[0, 3], 0] - self.xmin) <= self.tol)
        if edge == "right":
            return np.all(np.abs(xy_e[[1, 2], 0] - self.xmax) <= self.tol)
        if edge == "bottom":
            return np.all(np.abs(xy_e[[0, 1], 1] - self.ymin) <= self.tol)
        if edge == "top":
            return np.all(np.abs(xy_e[[2, 3], 1] - self.ymax) <= self.tol)
        return False

    def assembly(self):
        gps, w = gauss2x2()
        for e in range(self.Ne):
            nodes = self.conn[e, :]
            xy_e = self.xy[nodes, :]
            Ke = np.zeros((8, 8), dtype=float)
            fe = np.zeros(8, dtype=float)

            # area integration (stiffness + body force)
            for ig in range(4):
                xi, eta = gps[ig]
                N, dN_dxi, dN_deta = shape_quad4(xi, eta)
                _, detJ, dN_dxdy = jacobian_quad4(xy_e, dN_dxi, dN_deta)
                B = Bmat_quad4(dN_dxdy)
                Nmat = Nmat_quad4(N)
                Ke += (B.T @ self.D @ B) * detJ * self.t * w[ig]
                fe += (Nmat.T @ self.body_force) * detJ * self.t * w[ig]

            # traction loads on selected boundary edges
            for ld in self.loads:
                if ld.get("type") != "traction_edge":
                    continue
                edge = ld["edge"]
                tvec = np.asarray(ld["value"], dtype=float)
                if tvec.size != 2:
                    raise ValueError("traction_edge value must have length 2.")
                if self._element_on_boundary_edge(xy_e, edge):
                    fe += self._traction_on_element_edge(xy_e, edge, tvec)

            edofs = np.zeros(8, dtype=int)
            for a in range(4):
                edofs[2 * a] = 2 * nodes[a]
                edofs[2 * a + 1] = 2 * nodes[a] + 1

            self.Kglob[np.ix_(edofs, edofs)] += Ke
            self.Fglob[edofs] += fe

    def solve(self):
        all_dofs = np.arange(self.ndof, dtype=int)
        free_dofs = np.setdiff1d(all_dofs, self.constrained_dofs)
        if free_dofs.size == 0:
            raise ValueError("No free DOFs left after applying BCs.")

        Kff = self.Kglob[np.ix_(free_dofs, free_dofs)]
        Ff = self.Fglob[free_dofs].copy()
        if self.constrained_dofs.size > 0:
            Kfc = self.Kglob[np.ix_(free_dofs, self.constrained_dofs)]
            Ff -= Kfc @ self.constrained_vals

        try:
            Uf = np.linalg.solve(Kff, Ff)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "Singular system. Check boundary conditions for rigid-body modes."
            ) from exc

        self.U = np.zeros(self.ndof, dtype=float)
        self.U[free_dofs] = Uf
        if self.constrained_dofs.size > 0:
            self.U[self.constrained_dofs] = self.constrained_vals

    def postprocess(self):
        self.Rglob = self.Kglob @ self.U - self.Fglob
        self.reactions = np.zeros(self.ndof, dtype=float)
        if self.constrained_dofs.size > 0:
            self.reactions[self.constrained_dofs] = self.Rglob[self.constrained_dofs]

        self.U_nodes = self.U.reshape(self.Nn, 2)
        self.u_mag = np.linalg.norm(self.U_nodes, axis=1)
        self.max_u = float(np.max(self.u_mag))

        # Element stress at centroid (xi=0, eta=0)
        self.element_stress = np.zeros((self.Ne, 3), dtype=float)  # [sx, sy, txy]
        self.element_vm = np.zeros(self.Ne, dtype=float)
        for e in range(self.Ne):
            nodes = self.conn[e, :]
            xy_e = self.xy[nodes, :]
            ue = np.zeros(8, dtype=float)
            for a in range(4):
                ue[2 * a] = self.U[2 * nodes[a]]
                ue[2 * a + 1] = self.U[2 * nodes[a] + 1]
            _, dN_dxi, dN_deta = shape_quad4(0.0, 0.0)
            _, _, dN_dxdy = jacobian_quad4(xy_e, dN_dxi, dN_deta)
            B = Bmat_quad4(dN_dxdy)
            sig = self.D @ (B @ ue)
            self.element_stress[e, :] = sig
            sx, sy, txy = sig
            self.element_vm[e] = np.sqrt(sx**2 - sx * sy + sy**2 + 3.0 * txy**2)

        # Nodal averaged stress fields for contouring
        self.nodal_stress = self._nodal_average_stress(self.element_stress)
        self.nodal_vm = self._nodal_average_scalar(self.element_vm)

    def _nodal_average_stress(self, elem_sig):
        s = np.zeros((self.Nn, 3), dtype=float)
        c = np.zeros(self.Nn, dtype=int)
        for e in range(self.Ne):
            nodes = self.conn[e, :]
            for n in nodes:
                s[n, :] += elem_sig[e, :]
                c[n] += 1
        c = np.maximum(c, 1)
        return s / c[:, None]

    def _nodal_average_scalar(self, elem_sc):
        s = np.zeros(self.Nn, dtype=float)
        c = np.zeros(self.Nn, dtype=int)
        for e in range(self.Ne):
            nodes = self.conn[e, :]
            for n in nodes:
                s[n] += elem_sc[e]
                c[n] += 1
        c = np.maximum(c, 1)
        return s / c

    def results(self):
        return {
            "U": self.U,
            "U_nodes": self.U_nodes,
            "u_mag": self.u_mag,
            "max_u": self.max_u,
            "reactions": self.reactions,
            "constrained_dofs": self.constrained_dofs,
            "element_stress": self.element_stress,
            "element_vm": self.element_vm,
            "nodal_stress": self.nodal_stress,
            "nodal_vm": self.nodal_vm,
        }
