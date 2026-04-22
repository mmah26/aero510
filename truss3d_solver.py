import numpy as np


DOF_MAP = {"ux": 0, "uy": 1, "uz": 2}

def _as_array_per_element(value, n_elem):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_elem, float(arr), dtype=float)
    if arr.shape == (n_elem,):
        return arr.astype(float)
    raise ValueError(f"Expected scalar or shape ({n_elem},), got {arr.shape}.")


def _build_dof_table(n_nodes):
    return np.arange(3 * n_nodes, dtype=int).reshape(n_nodes, 3)


def _parse_constraints(constraints, dofs):
    c_dofs = []
    c_vals = []
    for bc in constraints:
        node = int(bc["node"])
        dof = bc["dof"]
        if dof not in DOF_MAP:
            raise ValueError(f'Unsupported dof "{dof}". Use ux, uy, uz.')
        c_dofs.append(dofs[node, DOF_MAP[dof]])
        c_vals.append(float(bc.get("value", 0.0)))
    if len(c_dofs) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    c_dofs = np.asarray(c_dofs, dtype=int)
    c_vals = np.asarray(c_vals, dtype=float)
    uniq, idx = np.unique(c_dofs, return_index=True)
    return uniq, c_vals[idx]


def _apply_point_loads(loads, dofs, n_dof):
    F = np.zeros(n_dof, dtype=float)
    for ld in loads:
        node = int(ld["node"])
        p = np.asarray(ld["value"], dtype=float)
        if p.shape != (3,):
            raise ValueError(f"Point load at node {node} must be length-3 [Fx,Fy,Fz].")
        F[dofs[node]] += p
    return F


def assemble_global_stiffness(V, E2N, E, A):
    n_nodes = V.shape[0]
    n_elem = E2N.shape[0]
    dofs = _build_dof_table(n_nodes)
    n_dof = 3 * n_nodes
    K = np.zeros((n_dof, n_dof), dtype=float)

    Es = _as_array_per_element(E, n_elem)
    As = _as_array_per_element(A, n_elem)

    p1 = V[E2N[:, 0]]
    p2 = V[E2N[:, 1]]
    d = p2 - p1
    L = np.linalg.norm(d, axis=1)
    if np.any(L <= 0.0):
        bad = np.where(L <= 0.0)[0]
        raise ValueError(f"Zero-length element(s) at indices: {bad}.")
    n = d / L[:, None]

    for i in range(n_elem):
        n1, n2 = E2N[i]
        edofs = dofs[[n1, n2]].ravel()
        k11 = (Es[i] * As[i] / L[i]) * np.outer(n[i], n[i])
        k_local = np.block([[k11, -k11], [-k11, k11]])
        K[np.ix_(edofs, edofs)] += k_local

    return K, L, n


def solve_truss_case(V, E2N, E, A, loads, constraints):
    n_nodes = V.shape[0]
    n_elem = E2N.shape[0]
    dofs = _build_dof_table(n_nodes)
    n_dof = 3 * n_nodes

    K, L, n = assemble_global_stiffness(V, E2N, E, A)
    F = _apply_point_loads(loads, dofs, n_dof)
    c_dofs, c_vals = _parse_constraints(constraints, dofs)

    all_dofs = np.arange(n_dof, dtype=int)
    free_dofs = np.setdiff1d(all_dofs, c_dofs)
    if free_dofs.size == 0:
        raise ValueError("No free DOFs left after constraints.")

    Kff = K[np.ix_(free_dofs, free_dofs)]
    Ff = F[free_dofs].copy()
    if c_dofs.size > 0:
        Kfc = K[np.ix_(free_dofs, c_dofs)]
        Ff -= Kfc @ c_vals

    try:
        Uf = np.linalg.solve(Kff, Ff)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "Singular system. Add sufficient supports/constraints for this load case."
        ) from exc

    U = np.zeros(n_dof, dtype=float)
    U[free_dofs] = Uf
    if c_dofs.size > 0:
        U[c_dofs] = c_vals

    # Reactions at constrained DOFs
    R = K @ U - F
    reactions = np.zeros_like(R)
    reactions[c_dofs] = R[c_dofs]

    # Element strain/stress
    Es = _as_array_per_element(E, n_elem)
    strain = np.zeros(n_elem, dtype=float)
    stress = np.zeros(n_elem, dtype=float)
    for i in range(n_elem):
        n1, n2 = E2N[i]
        u1 = U[dofs[n1]]
        u2 = U[dofs[n2]]
        axial_extension = np.dot(n[i], (u2 - u1))
        strain[i] = axial_extension / L[i]
        stress[i] = Es[i] * strain[i]

    U_nodes = U.reshape(n_nodes, 3)
    u_mag = np.linalg.norm(U_nodes, axis=1)

    return {
        "U": U,
        "U_nodes": U_nodes,
        "u_mag": u_mag,
        "max_u": float(np.max(u_mag)),
        "stress": stress,
        "strain": strain,
        "reactions": reactions,
        "constrained_dofs": c_dofs,
    }
