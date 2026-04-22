import matplotlib.pyplot as plt
from pathlib import Path

from hex_geometry import E_al, hex_geometry, truss_areas
from truss3d_solver import DirectFEM3D, fix_node_all_dofs, fix_node_dofs
from postproc_truss3d import (
    summarize_case,
    deformation_scale,
    write_case_outputs,
    plot_truss_with_bcs_loads_3d,
    plot_deformed_stress_truss_3d,
)

# Nonplanar geometry is required for truss z-load response.
# Here: center hub elevated by 6 in relative to rim nodes.
V, E2N = hex_geometry(hub_z=0.5, rim_z=0.0)
A = truss_areas(E2N)
E = E_al
out_dir = Path(__file__).resolve().parent / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

# Case dictionaries in a sample_case.py-like structure:
# mesh/properties/loads/bcs can be swapped cleanly per scenario.
case1 = {
    "name": "Case 1: Mid-flight package load",
    "mesh": {"V": V, "E2N": E2N},
    "properties": {"E": E, "A": A},
    "loads": [{"type": "point", "node": 0, "value": [0.0, 0.0, -5.0]}],
    "bcs": (
        fix_node_all_dofs(1)
        + fix_node_all_dofs(2)
        + fix_node_all_dofs(3)
        + fix_node_all_dofs(4)
        + fix_node_all_dofs(5)
        + fix_node_all_dofs(6)
    ),
}

case2 = {
    "name": "Case 2: Child pull at node 1, adult hold at node 4",
    "mesh": {"V": V, "E2N": E2N},
    "properties": {"E": E, "A": A},
    "loads": [{"type": "point", "node": 1, "value": [15.0, 0.0, 0.0]}],
    "bcs": (
        fix_node_all_dofs(4)
        + fix_node_dofs(0, ["uy", "uz"])
        + fix_node_dofs(1, ["uz"])
    ),
}

case3_load_nodes = [0, 1, 2, 3, 4, 6]
case3 = {
    "name": "Case 3: Tangled drone, distributed thrust",
    "mesh": {"V": V, "E2N": E2N},
    "properties": {"E": E, "A": A},
    "loads": [{"type": "point", "node": n, "value": [0.0, 0.0, 15.0]} for n in case3_load_nodes],
    "bcs": (
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
        fem = DirectFEM3D(case=case)
        out = fem.results()
        summarize_case(case["name"], out)
        write_case_outputs(out_dir, case_tags[i], case["mesh"]["E2N"], out)

        scale = deformation_scale(V, out["U_nodes"])

        fig1 = plt.figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111, projection="3d")
        plot_truss_with_bcs_loads_3d(
            V,
            E2N,
            bcs=case["bcs"],
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