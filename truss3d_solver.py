import numpy as np
from dataclasses import dataclass

# Local dof-key -> local dof-index map at a node.
# Used when parsing BC dictionaries (e.g., "uy" means local index 1).
DOF_MAP = {"ux": 0, "uy": 1, "uz": 2}

def prop_array_per_element(value, n_elem):
    """
    Normalize material/property input to per-element array.
    - If scalar is provided: broadcast to length n_elem.
    - If array of shape (n_elem,) is provided: use directly.
    """
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_elem, float(arr), dtype=float)
    if arr.shape == (n_elem,):
        return arr.astype(float)
    raise ValueError(f"Expected scalar or shape ({n_elem},), got {arr.shape}.")

def fix_node_all_dofs(node, value=0.0):
    """
    Build BC dictionaries to constrain all 3 translational dofs of one node.
    """
    return [
        {"type": "dirichlet", "node": node, "dof": "ux", "value": float(value)},
        {"type": "dirichlet", "node": node, "dof": "uy", "value": float(value)},
        {"type": "dirichlet", "node": node, "dof": "uz", "value": float(value)},
    ]

def fix_node_dofs(node, dofs, value=0.0):
    """
    Build BC dictionaries to constrain only selected dofs of one node.
    Example: fix_node_dofs(0, ["uy", "uz"])
    """
    return [{"type": "dirichlet", "node": node, "dof": d, "value": float(value)} for d in dofs]

@dataclass
class DirectFEM3D:
    case: dict

    def __post_init__(self):
        # Parse case dictionary
        self.V = np.asarray(self.case["mesh"]["V"], dtype=float)
        self.E2N = np.asarray(self.case["mesh"]["E2N"], dtype=int)
        if self.V.shape[1] != 3:
            raise ValueError(f"DirectFEM3D expects 3D coordinates [N,3], got {self.V.shape}.")

        # Convert 1-based connectivity to 0-based if needed.
        if self.E2N.min() == 0:
            pass
        elif self.E2N.min() == 1:
            self.E2N = self.E2N - 1
        else:
            raise ValueError("E2N indexing must be either 0-based or 1-based.")

        # Properties (scalar or per-element arrays)
        self.nne = self.E2N.shape[1]    # number of nodes per element
        self.ndim = self.V.shape[1]     # number of spatial dimensions (by size of coordinate array)
        self.Ne = self.E2N.shape[0]     # number of elements
        self.Nn = self.V.shape[0]       # number of nodes 
        self.ndof = self.Nn * self.ndim # total degrees of freedom = nodes * spatial dof

        self.Es = prop_array_per_element(self.case["properties"]["E"], self.Ne)
        self.As = prop_array_per_element(self.case["properties"]["A"], self.Ne)

        # Global arrays
        self.Kglob = np.zeros((self.ndof, self.ndof), dtype=float)
        self.Fglob = np.zeros(self.ndof, dtype=float)
        self.dofs = np.arange(self.ndof, dtype=int).reshape(self.Nn, self.ndim)

        # Geometry per element (3D)
        p1 = self.V[self.E2N[:, 0]]
        p2 = self.V[self.E2N[:, 1]]
        d = p2 - p1
        self.Ls = np.linalg.norm(d, axis=1)
        if np.any(self.Ls <= 0.0):
            bad = np.where(self.Ls <= 0.0)[0]
            raise ValueError(f"Zero-length element(s) at indices: {bad}.")
        self.ns = d / self.Ls[:, None]  # direction cosines in 3D

        # Loads
        for load in self.case.get("loads", []):
            ltype = load.get("type", "point")
            if ltype != "point":
                continue
            node = int(load["node"])
            p = np.asarray(load["value"], dtype=float)
            if p.size != self.ndim:
                raise ValueError(f"Point load at node {node} must have {self.ndim} components.")
            self.Fglob[self.dofs[node]] += p

        # Displacement BCs
        c_dofs = []
        c_vals = []
        for bc in self.case.get("bcs", []):
            btype = bc.get("type", "dirichlet")
            if btype != "dirichlet":
                continue
            node = int(bc["node"])
            dof_key = bc["dof"]
            if dof_key not in DOF_MAP:
                raise ValueError(f'Unsupported BC dof "{dof_key}". Use ux, uy, uz.')
            gdof = self.dofs[node, DOF_MAP[dof_key]]
            c_dofs.append(gdof)
            c_vals.append(float(bc.get("value", 0.0)))

        if len(c_dofs) == 0:
            self.constrained_dofs = np.array([], dtype=int)
            self.constrained_vals = np.array([], dtype=float)
        else:
            c_dofs = np.asarray(c_dofs, dtype=int)
            c_vals = np.asarray(c_vals, dtype=float)
            uniq, idx = np.unique(c_dofs, return_index=True)
            self.constrained_dofs = uniq
            self.constrained_vals = c_vals[idx]

        self.assembly()
        self.solve()
        self.postprocess()

    def assembly(self):
        for i in range(self.Ne):
            n1, n2 = self.E2N[i, :]
            edofs = self.dofs[[n1, n2]].ravel()

            k11 = self.Es[i] * self.As[i] / self.Ls[i] * np.outer(self.ns[i], self.ns[i])
            kloc = np.block([[k11, -k11], [-k11, k11]])
            floc = np.zeros(edofs.shape[0], dtype=float)

            self.Kglob[np.ix_(edofs, edofs)] += kloc
            self.Fglob[edofs] += floc

    def solve(self):
        all_dofs = np.arange(self.ndof, dtype=int)
        free_dofs = np.setdiff1d(all_dofs, self.constrained_dofs)
        if free_dofs.size == 0:
            raise ValueError("No free DOFs left after applying boundary conditions.")

        Kff = self.Kglob[np.ix_(free_dofs, free_dofs)]
        Ff = self.Fglob[free_dofs].copy()
        if self.constrained_dofs.size > 0:
            Kfc = self.Kglob[np.ix_(free_dofs, self.constrained_dofs)]
            Ff -= Kfc @ self.constrained_vals

        try:
            Uf = np.linalg.solve(Kff, Ff)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "Singular system. Add sufficient supports/constraints for this load case."
            ) from exc

        self.U = np.zeros(self.ndof, dtype=float)
        self.U[free_dofs] = Uf
        if self.constrained_dofs.size > 0:
            self.U[self.constrained_dofs] = self.constrained_vals

    def postprocess(self):
        # Reactions
        self.Rglob = self.Kglob @ self.U - self.Fglob
        self.reactions = np.zeros(self.ndof, dtype=float)
        if self.constrained_dofs.size > 0:
            self.reactions[self.constrained_dofs] = self.Rglob[self.constrained_dofs]

        # Element strain/stress
        self.strain = np.zeros(self.Ne, dtype=float)
        self.stress = np.zeros(self.Ne, dtype=float)
        for i in range(self.Ne):
            n1, n2 = self.E2N[i, :]
            u1 = self.U[self.dofs[n1]]
            u2 = self.U[self.dofs[n2]]
            axial_extension = np.dot(self.ns[i], (u2 - u1))
            self.strain[i] = axial_extension / self.Ls[i]
            self.stress[i] = self.Es[i] * self.strain[i]

        self.U_nodes = self.U.reshape(self.Nn, self.ndim)
        self.u_mag = np.linalg.norm(self.U_nodes, axis=1)
        self.max_u = float(np.max(self.u_mag))

    def results(self):
        return {
            "U": self.U,
            "U_nodes": self.U_nodes,
            "u_mag": self.u_mag,
            "max_u": self.max_u,
            "stress": self.stress,
            "strain": self.strain,
            "reactions": self.reactions,
            "constrained_dofs": self.constrained_dofs,
        }


# def solve_truss_case(V, E2N, E, A, loads, constraints):
#     # Compatibility wrapper for existing call sites.
#     case = {
#         "mesh": {"V": np.asarray(V, dtype=float), "E2N": np.asarray(E2N, dtype=int)},
#         "properties": {"E": E, "A": A},
#         "loads": [{"type": "point", **ld} if "type" not in ld else ld for ld in loads],
#         "bcs": [{"type": "dirichlet", **bc} if "type" not in bc else bc for bc in constraints],
#     }
#     fem = DirectFEM3D(case=case)
#     return fem.results()
