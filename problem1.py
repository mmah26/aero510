import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from hex_geometry import E_al, hex_geometry, truss_areas
from truss3d_solver import solve_truss_case
from postproc_truss3d import (
    plot_truss_with_bcs_loads_3d,
    plot_deformed_stress_truss_3d,
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


def write_case_outputs(out_dir, case_tag, E2N, out):
    # Node displacement output
    n_nodes = out["U_nodes"].shape[0]
    node_ids = np.arange(n_nodes, dtype=int)
    node_table = np.column_stack([node_ids, out["U_nodes"], out["u_mag"]])
    node_header = "node,ux_ft,uy_ft,uz_ft,u_mag_ft"
    np.savetxt(
        out_dir / f"{case_tag}_node_displacements.csv",
        node_table,
        delimiter=",",
        header=node_header,
        comments="",
        fmt=["%d", "%.8e", "%.8e", "%.8e", "%.8e"],
    )

    # Element strain/stress output
    elem_ids = np.arange(E2N.shape[0], dtype=int)
    elem_table = np.column_stack([elem_ids, E2N[:, 0], E2N[:, 1], out["strain"], out["stress"]])
    elem_header = "element,node_i,node_j,strain,stress_pa"
    np.savetxt(
        out_dir / f"{case_tag}_element_results.csv",
        elem_table,
        delimiter=",",
        header=elem_header,
        comments="",
        fmt=["%d", "%d", "%d", "%.8e", "%.8e"],
    )

    # Short summary file
    with open(out_dir / f"{case_tag}_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"max_nodal_deflection_ft={out['max_u']:.8e}\n")
        f.write(f"min_element_stress_pa={np.min(out['stress']):.8e}\n")
        f.write(f"max_element_stress_pa={np.max(out['stress']):.8e}\n")
        f.write(f"max_abs_element_stress_pa={np.max(np.abs(out['stress'])):.8e}\n")


def main():
    # Nonplanar geometry is required for truss z-load response.
    # Here: center hub elevated by 6 in relative to rim nodes.
    V, E2N = hex_geometry(hub_z=0.5, rim_z=0.0)
    A = truss_areas(E2N)
    E = E_al
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

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
    case_tags = ["case1", "case2", "case3"]
    case_titles = ["Case 1", "Case 2", "Case 3"]
    print("Using nonplanar 3D hex geometry: hub z = +0.5 ft, rim z = 0 ft.")

    for i, case in enumerate(cases):
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
            write_case_outputs(out_dir, case_tags[i], E2N, out)

            scale = deformation_scale(V, out["U_nodes"])

            fig1 = plt.figure(figsize=(8, 6))
            ax1 = fig1.add_subplot(111, projection="3d")
            plot_truss_with_bcs_loads_3d(
                V,
                E2N,
                bcs=case["constraints"],
                loads=case["loads"],
                ax=ax1,
                title=case_titles[i],
            )
            fig1.subplots_adjust(left=0.06, right=0.93, bottom=0.08, top=0.92)
            fig1.savefig(out_dir / f"{case_tags[i]}_setup_undeformed.png", dpi=220)

            fig2 = plt.figure(figsize=(8, 6))
            ax2 = fig2.add_subplot(111, projection="3d")
            plot_deformed_stress_truss_3d(
                V,
                E2N,
                U_nodes=out["U_nodes"],
                stress=out["stress"],
                scale=scale,
                include_undeformed=True,
                ax=ax2,
                title=case_titles[i],
            )
            fig2.subplots_adjust(left=0.06, right=0.90, bottom=0.08, top=0.92)
            fig2.savefig(out_dir / f"{case_tags[i]}_deform_stress.png", dpi=220)
        except np.linalg.LinAlgError as exc:
            print(f"\n{case['name']}")
            print("  solve failed: singular stiffness matrix")
            print(f"  reason: {exc}")
            print("  note: this support/load definition is kinematically unstable for the current geometry.")

    plt.close("all")


if __name__ == "__main__":
    main()
