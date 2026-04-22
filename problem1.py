import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from hex_geometry import E_al, hex_geometry, truss_areas
from truss3d_solver import DirectFEM3D, fix_node_all_dofs, fix_node_dofs
from postproc_truss3d import (
    summarize_case,
    deformation_scale,
    write_case_outputs,
    plot_hex_2D,
    plot_truss_with_bcs_loads_3d,
    plot_deformed_stress_truss_3d,
)

# Nonplanar geometry is required for truss z-load response.
# center hub dropped by 2 in relative to rim nodes.
hub_z_shift = - 2 / 12 
V, E2N = hex_geometry(hub_z=hub_z_shift, rim_z=0.0)
A = truss_areas(E2N)
E = E_al
out_dir = Path(__file__).resolve().parent / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
plot_hex_2D(V, E2N, out_path=out_dir / "hex_2D.svg", show=False)

# Case dictionaries in a sample_case.py-like structure:
# mesh/properties/loads/bcs can be swapped cleanly per scenario.

# Case 1 assumptions (mid-flight package load):
# - Physical intent: payload acts downward at the hub while perimeter nodes are treated as rigidly supported.
# - Load model: point load at node 0 in -z.
# - BC rationale: nodes 1..6 fully fixed (ux=uy=uz=0) to represent a conservative, stiff support condition
#   and isolate local spoke response around the hub.
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

# Case 2 assumptions (child pull while adult holds opposite side):
# - Physical intent: pulling node 1 in +x while node 4 is held fixed.
# - Load model: point load +225 lbf in +x at node 1.
# - BC rationale:
#   1) node 4 fully fixed = holding point,
#   2) node 1: uy=uz=0 to keep applied pull directional in +x,
#   3) node 0: uy=0 to suppress residual rigid-body in-plane mode and keep a stable solve.
case2 = {
    "name": "Case 2: Child pull at node 1, adult hold at node 4",
    "mesh": {"V": V, "E2N": E2N},
    "properties": {"E": E, "A": A},
    "loads": [{"type": "point", "node": 1, "value": [225.0, 0.0, 0.0]}],
    "bcs": (
        fix_node_all_dofs(4)
        + fix_node_dofs(1, ["uy", "uz"])
        + fix_node_dofs(0, ["uy"])
        + fix_node_dofs(2, ["uz"])
        + fix_node_dofs(3, ["uz"])
        + fix_node_dofs(5, ["uz"])
        + fix_node_dofs(6, ["uz"])
    ),
}

# Case 3 assumptions (vehicle snagged at node 5, thrusting upward):
# - Physical intent: node 5 is tangled/anchored, remaining nodes supply upward thrust.
# - Load model: +15 lbf in +z at nodes 0,1,2,3,4,6.
# - BC rationale:
#   1) node 5 fully fixed = snag/contact point,
#   2) nodes 0,1,2,3,4,6 have ux=uy=0 so motion is restricted to vertical response (z-direction thrust model),
#      representing an idealized "no in-plane drift" assumption while attempting lift-off.
case3_load_nodes = [0, 1, 2, 3, 4, 6]
case3 = {
    "name": "Case 3: Tangled drone, distributed thrust",
    "mesh": {"V": V, "E2N": E2N},
    "properties": {"E": E, "A": A},
    "loads": [{"type": "point", "node": n, "value": [0.0, 0.0, 15.0]} for n in case3_load_nodes],
    "bcs": (
        fix_node_all_dofs(5)
        + fix_node_dofs(0, ["ux", "uy"])
        + fix_node_dofs(1, ["ux", "uy"])
        + fix_node_dofs(2, ["ux", "uy"])
        + fix_node_dofs(3, ["ux", "uy"])
        + fix_node_dofs(4, ["ux", "uy"])
        + fix_node_dofs(6, ["ux", "uy"])
    ),
}

cases = [case1, case2, case3]
case_tags = ["case1", "case2", "case3"]
case_titles = ["Case 1", "Case 2", "Case 3"]
print(f"Using nonplanar 3D hex geometry: hub z = {hub_z_shift:.4f} ft, rim z = 0 ft.")

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
