import numpy as np
import matplotlib.pyplot as plt

from hex_geometry import E_AL_PA, get_hex_geometry_ft, get_member_areas_ft2
from truss3d_solver import solve_truss_case
from util3d import (
    plot_truss_with_bcs_loads_3d,
    plot_deformed_truss_3d,
    plot_stress_truss_3d,
)


def fix_node_all_dofs(node):
    return [
        {"node": node, "dof": "ux", "value": 0.0},
        {"node": node, "dof": "uy", "value": 0.0},
        {"node": node, "dof": "uz", "value": 0.0},
    ]


def fix_node_dofs(node, dofs):
    return [{"node": node, "dof": d, "value": 0.0} for d in dofs]


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


def main():
    # Nonplanar geometry is required for truss z-load response.
    # Here: center hub elevated by 6 in relative to rim nodes.
    V, E2N = get_hex_geometry_ft(hub_z_ft=0.5, rim_z_ft=0.0)
    A = get_member_areas_ft2(E2N)
    E = E_AL_PA

    # Loading case 1:
    # 5 lbf at node 0 in -z; all other nodes fixed.
    case1 = {
        "name": "Case 1: Mid-flight package load",
        "loads": [{"node": 0, "value": [0.0, 0.0, -5.0]}],
        "constraints": (
            fix_node_all_dofs(1)
            + fix_node_all_dofs(2)
            + fix_node_all_dofs(3)
            + fix_node_all_dofs(4)
            + fix_node_all_dofs(5)
            + fix_node_all_dofs(6)
        ),
    }

    # Loading case 2:
    # 15 lbf at node 1 in +x; node 4 fixed.
    case2 = {
        "name": "Case 2: Child pull at node 1, adult hold at node 4",
        "loads": [{"node": 1, "value": [15.0, 0.0, 0.0]}],
        "constraints": (
            fix_node_all_dofs(4)
            + fix_node_dofs(0, ["uy", "uz"])
            + fix_node_dofs(1, ["uz"])
        ),
    }

    # Loading case 3:
    # Node 5 fixed; 15 lbf at nodes 0,1,2,3,4,6 in +z.
    case3_load_nodes = [0, 1, 2, 3, 4, 6]
    case3 = {
        "name": "Case 3: Tangled drone, distributed thrust",
        "loads": [{"node": n, "value": [0.0, 0.0, 15.0]} for n in case3_load_nodes],
        "constraints": (
            fix_node_all_dofs(5)
            + fix_node_dofs(0, ["uy", "uz"])
            + fix_node_dofs(1, ["uz"])
        ),
    }

    cases = [case1, case2, case3]
    print("Using nonplanar 3D hex geometry: hub z = +0.5 ft, rim z = 0 ft.")

    for case in cases:
        try:
            out = solve_truss_case(
                V=V,
                E2N=E2N,
                E=E,
                A=A,
                loads=case["loads"],
                constraints=case["constraints"],
            )
            summarize_case(case["name"], out)

            scale = deformation_scale(V, out["U_nodes"])

            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111, projection="3d")
            plot_truss_with_bcs_loads_3d(
                V,
                E2N,
                bcs=case["constraints"],
                loads=case["loads"],
                ax=ax1,
                title=f"{case['name']} - Loads/BCs",
            )

            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111, projection="3d")
            plot_deformed_truss_3d(
                V,
                E2N,
                U_nodes=out["U_nodes"],
                scale=scale,
                ax=ax2,
                title=f"{case['name']} - Deformed (scale={scale:.2e})",
            )

            fig3 = plt.figure()
            ax3 = fig3.add_subplot(111, projection="3d")
            plot_stress_truss_3d(
                V,
                E2N,
                stress=out["stress"],
                ax=ax3,
                title=f"{case['name']} - Axial Stress",
            )
        except np.linalg.LinAlgError as exc:
            print(f"\n{case['name']}")
            print("  solve failed: singular stiffness matrix")
            print(f"  reason: {exc}")
            print("  note: this support/load definition is kinematically unstable for the current geometry.")

    plt.show()


if __name__ == "__main__":
    main()
