from pathlib import Path
import numpy as np

from panel_geometry import L_M, W_M, T_M, E_AL, NU_AL, gen_quad_mesh
from galerkinFEM_quad import DirectFEMQuad, dirichlet_bcs_for_edge, traction_load
from postproc_galerkinFEM import (
    summarize_case,
    write_case_outputs,
    save_case_plots,
    plot_case_convergence,
)


def build_case_definitions(xy, t, tmag):
    """
    Return case definitions for a given mesh.
    tmag is line traction in N/m.
    """
    case1 = {
        "name": "Case 1",
        "mesh": {"xy": xy, "conn": None},  # conn assigned by caller
        "properties": {"E": E_AL, "nu": NU_AL, "t": t, "plane": "stress"},
        "body_force": [0.0, 0.0],
        "loads": [traction_load("right", [tmag, 0.0])],
        "bcs": dirichlet_bcs_for_edge(xy, "left", dofs=("ux", "uy"), value=0.0),
    }

    case2 = {
        "name": "Case 2",
        "mesh": {"xy": xy, "conn": None},
        "properties": {"E": E_AL, "nu": NU_AL, "t": t, "plane": "stress"},
        "body_force": [0.0, 0.0],
        "loads": [traction_load("right", [tmag / np.sqrt(2.0), tmag / np.sqrt(2.0)])],
        "bcs": dirichlet_bcs_for_edge(xy, "left", dofs=("ux", "uy"), value=0.0),
    }

    case3 = {
        "name": "Case 3",
        "mesh": {"xy": xy, "conn": None},
        "properties": {"E": E_AL, "nu": NU_AL, "t": t, "plane": "stress"},
        "body_force": [0.0, 0.0],
        "loads": [traction_load("bottom", [0.0, -tmag])],
        "bcs": (
            dirichlet_bcs_for_edge(xy, "left", dofs=("ux", "uy"), value=0.0)
            + dirichlet_bcs_for_edge(xy, "right", dofs=("ux", "uy"), value=0.0)
        ),
    }
    return [case1, case2, case3]


def main():
    out_dir = Path(__file__).resolve().parent / "figures" / "panel"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Three target mesh sizes: ~10, ~100, ~1000 elements.
    # Using exact products nx*ny = 10, 100, 1000.
    mesh_levels = [
        {"label": "n10", "nx": 5, "ny": 2},
        {"label": "n100", "nx": 20, "ny": 5},
        {"label": "n1000", "nx": 50, "ny": 20},
    ]

    case_tags = ["case1", "case2", "case3"]
    case_titles = ["Case 1", "Case 2", "Case 3"]
    conv = {k: {"ne": [], "sx": [], "sy": [], "svm": []} for k in case_tags}

    # Edge traction as line load (N/m): equivalent to 1 MPa pressure over thickness.
    p_ref = 1.0e6  # Pa
    tmag = p_ref * T_M

    for lvl in mesh_levels:
        xy, conn = gen_quad_mesh(L_M, W_M, lvl["nx"], lvl["ny"])
        ne = int(conn.shape[0])
        cases = build_case_definitions(xy, T_M, tmag)

        print(f"\n=== Mesh {lvl['label']} ({lvl['nx']}x{lvl['ny']} => {ne} elements) ===")
        for i, case in enumerate(cases):
            case["mesh"]["conn"] = conn
            run_tag = f"{case_tags[i]}_{lvl['label']}"
            case_run = dict(case)
            case_run["name"] = f"{case_titles[i]} ({ne} elems)"

            fem = DirectFEMQuad(case=case_run)
            out = fem.results()
            summarize_case(case_run["name"], out)
            write_case_outputs(out_dir, run_tag, xy, conn, out)
            save_case_plots(out_dir, run_tag, case_run, out)

            conv[case_tags[i]]["ne"].append(ne)
            conv[case_tags[i]]["sx"].append(float(np.max(np.abs(out["element_stress"][:, 0]))))
            conv[case_tags[i]]["sy"].append(float(np.max(np.abs(out["element_stress"][:, 1]))))
            conv[case_tags[i]]["svm"].append(float(np.max(out["element_vm"])))

    # Convergence plots: one per case, 3 curves (sx, sy, svm), 3 mesh points.
    for tag, title in zip(case_tags, case_titles):
        ne = np.asarray(conv[tag]["ne"], dtype=int)
        order = np.argsort(ne)
        plot_case_convergence(
            out_dir=out_dir,
            case_tag=tag,
            case_title=title,
            elem_counts=ne[order],
            sx_vals=np.asarray(conv[tag]["sx"])[order],
            sy_vals=np.asarray(conv[tag]["sy"])[order],
            svm_vals=np.asarray(conv[tag]["svm"])[order],
        )


if __name__ == "__main__":
    main()
