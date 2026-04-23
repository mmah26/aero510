from pathlib import Path
import numpy as np

from panel_geometry import L_M, W_M, T_M, E_AL, NU_AL, gen_quad_mesh
from galerkinFEM_quad import DirectFEMQuad, dirichlet_bcs_for_edge, traction_load
from postproc_galerkinFEM import summarize_case, write_case_outputs, save_case_plots

out_dir = Path(__file__).resolve().parent / "figures" / "panel"
out_dir.mkdir(parents=True, exist_ok=True)

L, W, t = L_M, W_M, T_M

# Structured square-like panel mesh (quad elements)
nx, ny = 24, 6
xy, conn = gen_quad_mesh(L, W, nx, ny)

# Edge traction magnitude used for all cases as line load (N/m).
# Equivalent to 1 MPa pressure acting over panel thickness: q = p * t.
p_ref = 1.0e6  # Pa
tmag = p_ref * t  # N/m

"""
Case 1:
Uniform traction in the +x direction (x-axis tension on the right)
boundary condition: fix the left edge (ux, uy = 0)
"""
case1 = {
    "name": "Case 1",
    "mesh": {"xy": xy, "conn": conn},
    "properties": {"E": E_AL, "nu": NU_AL, "t": t, "plane": "stress"},
    "body_force": [0.0, 0.0],
    "loads": [traction_load("right", [tmag, 0.0])],
    "bcs": dirichlet_bcs_for_edge(xy, "left", dofs=("ux", "uy"), value=0.0),
}

"""
Case 2:
Traction on right edge up and to the right (+45 deg angle)
boundary condition: fix the left edge (ux, uy = 0)
"""
case2 = {
    "name": "Case 2",
    "mesh": {"xy": xy, "conn": conn},
    "properties": {"E": E_AL, "nu": NU_AL, "t": t, "plane": "stress"},
    "body_force": [0.0, 0.0],
    "loads": [traction_load("right", [tmag / np.sqrt(2.0), tmag / np.sqrt(2.0)])],
    "bcs": dirichlet_bcs_for_edge(xy, "left", dofs=("ux", "uy"), value=0.0),
}

"""
Case 3:
Uniform traction on bottom edge -y direction
boundary condition: fix the left edge (ux, uy = 0) and the right edge
"""
case3 = {
    "name": "Case 3",
    "mesh": {"xy": xy, "conn": conn},
    "properties": {"E": E_AL, "nu": NU_AL, "t": t, "plane": "stress"},
    "body_force": [0.0, 0.0],
    "loads": [traction_load("bottom", [0.0, -tmag])],
    "bcs": (
        dirichlet_bcs_for_edge(xy, "left", dofs=("ux", "uy"), value=0.0)
        + dirichlet_bcs_for_edge(xy, "right", dofs=("ux", "uy"), value=0.0)
    ),
}

cases = [case1, case2, case3]
case_tags = ["case1", "case2", "case3"]

for case, tag in zip(cases, case_tags):
    fem = DirectFEMQuad(case=case)
    out = fem.results()
    summarize_case(case["name"], out)
    write_case_outputs(out_dir, tag, xy, conn, out)
    save_case_plots(out_dir, tag, case, out)