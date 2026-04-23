from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from panel_geometry import L_M, W_M, T_M, gen_quad_mesh
from cfrp_lamina import cfrp_ud_effective_properties, traction_vector_from_angle
from galerkinFEM_quad import DirectFEMQuad, dirichlet_bcs_for_edge, traction_load
from postproc_galerkinFEM import (
    summarize_case,
    write_case_outputs,
    save_case_plots,
    plot_orientation_sweep,
)


def make_cfrp_case(
    name,
    xy,
    conn,
    t,
    traction_vec,
    orth_props,
    theta_deg,
):
    return {
        "name": name,
        "mesh": {"xy": xy, "conn": conn},
        "properties": {
            "E1": orth_props["E1"],
            "E2": orth_props["E2"],
            "G12": orth_props["G12"],
            "nu12": orth_props["nu12"],
            "theta_deg": float(theta_deg),
            "t": t,
            "plane": "stress",
        },
        "body_force": [0.0, 0.0],
        "loads": [traction_load("right", traction_vec)],
        "bcs": dirichlet_bcs_for_edge(xy, "left", dofs=("ux", "uy"), value=0.0),
    }


def write_problem3_summary(out_dir, orth, best_case1, best_arb, angle_rows):
    out_txt = out_dir / "problem3_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("CFRP Effective Orthotropic Properties (UD lamina)\n")
        f.write(f"E1_pa={orth['E1']:.8e}\n")
        f.write(f"E2_pa={orth['E2']:.8e}\n")
        f.write(f"G12_pa={orth['G12']:.8e}\n")
        f.write(f"nu12={orth['nu12']:.8e}\n\n")

        f.write("Recommended Fiber Orientation\n")
        f.write(
            f"case1_plus_x_theta_deg={best_case1['theta_deg']:.1f}, "
            f"max_sigma_vm_pa={best_case1['sigma_vm_pa']:.8e}\n"
        )
        f.write(
            f"arbitrary_cases_2_to_5_theta_deg={best_arb['theta_deg']:.1f}, "
            f"worst_case_sigma_vm_pa={best_arb['sigma_vm_pa']:.8e}\n"
        )

    csv = out_dir / "problem3_arbitrary_cases_vs_angle.csv"
    header = "theta_deg,case2_sigma_vm_pa,case3_sigma_vm_pa,case4_sigma_vm_pa,case5_sigma_vm_pa,worst_of_2_to_5_pa"
    np.savetxt(csv, np.asarray(angle_rows), delimiter=",", header=header, comments="", fmt="%.8e")


def main():
    out_dir = Path(__file__).resolve().parent / "figures" / "panel_composite"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use ~100 element mesh for CFRP orientation studies.
    nx, ny = 20, 5
    xy, conn = gen_quad_mesh(L_M, W_M, nx, ny)

    # Traction magnitude as line load: p*t, with p = 1 MPa reference.
    p_ref = 1.0e6  # Pa
    tmag = p_ref * T_M  # N/m

    # CFRP constituent properties from lecture slide.
    Ef = 241.0e9
    nuf = 0.2
    Em = 3.12e9
    num = 0.38
    Vf = 0.60

    orth = cfrp_ud_effective_properties(Ef=Ef, nuf=nuf, Em=Em, num=num, Vf=Vf)
    print("\nComputed CFRP UD effective properties (plane stress lamina):")
    print(f"  E1  = {orth['E1']:.6e} Pa")
    print(f"  E2  = {orth['E2']:.6e} Pa")
    print(f"  G12 = {orth['G12']:.6e} Pa")
    print(f"  nu12= {orth['nu12']:.6f}")

    # Five right-edge traction cases.
    load_cases = [
        ("case1", "Case 1 (+x traction)", 0.0),
        ("case2", "Case 2 (30 deg traction)", 30.0),
        ("case3", "Case 3 (45 deg traction)", 45.0),
        ("case4", "Case 4 (60 deg traction)", 60.0),
        ("case5", "Case 5 (+y traction)", 90.0),
    ]

    # Baseline output at fiber orientation theta = 0 deg.
    theta_ref = 0.0
    for tag, title, load_angle in load_cases:
        tvec = traction_vector_from_angle(tmag, load_angle)
        case = make_cfrp_case(
            name=f"{title}, fiber {theta_ref:.0f} deg",
            xy=xy,
            conn=conn,
            t=T_M,
            traction_vec=tvec,
            orth_props=orth,
            theta_deg=theta_ref,
        )
        fem = DirectFEMQuad(case=case)
        out = fem.results()
        summarize_case(case["name"], out)
        write_case_outputs(out_dir, f"{tag}_theta0", xy, conn, out)
        save_case_plots(out_dir, f"{tag}_theta0", case, out)

    # Fiber-angle sweep for recommendation and proof.
    angles = np.arange(0.0, 91.0, 5.0)
    metrics = {
        tag: {"sx": [], "sy": [], "svm": []}
        for tag, _, _ in load_cases
    }

    for theta in angles:
        for tag, title, load_angle in load_cases:
            tvec = traction_vector_from_angle(tmag, load_angle)
            case = make_cfrp_case(
                name=f"{title}, fiber {theta:.0f} deg",
                xy=xy,
                conn=conn,
                t=T_M,
                traction_vec=tvec,
                orth_props=orth,
                theta_deg=theta,
            )
            out = DirectFEMQuad(case=case).results()
            metrics[tag]["sx"].append(float(np.max(np.abs(out["element_stress"][:, 0]))))
            metrics[tag]["sy"].append(float(np.max(np.abs(out["element_stress"][:, 1]))))
            metrics[tag]["svm"].append(float(np.max(out["element_vm"])))

    # Plot stress-vs-fiber-angle curves (proof by sweep).
    for tag, title, _ in load_cases:
        plot_orientation_sweep(
            out_dir=out_dir,
            case_tag=tag,
            case_title=title,
            angles_deg=angles,
            sx_vals=np.asarray(metrics[tag]["sx"]),
            sy_vals=np.asarray(metrics[tag]["sy"]),
            svm_vals=np.asarray(metrics[tag]["svm"]),
        )

        table = np.column_stack(
            [
                angles,
                np.asarray(metrics[tag]["sx"]),
                np.asarray(metrics[tag]["sy"]),
                np.asarray(metrics[tag]["svm"]),
            ]
        )
        np.savetxt(
            out_dir / f"{tag}_orientation_sweep.csv",
            table,
            delimiter=",",
            header="theta_deg,max_abs_sigma_x_pa,max_abs_sigma_y_pa,max_sigma_vm_pa",
            comments="",
            fmt="%.8e",
        )

    # Recommendation 1: best angle for +x traction (case1) based on min max sigma_vm.
    case1_vm = np.asarray(metrics["case1"]["svm"])
    i1 = int(np.argmin(case1_vm))
    best_case1 = {"theta_deg": float(angles[i1]), "sigma_vm_pa": float(case1_vm[i1])}

    # Recommendation 2: robust angle for arbitrary tractions (cases 2..5), minimax on sigma_vm.
    vm_arb = np.vstack(
        [
            np.asarray(metrics["case2"]["svm"]),
            np.asarray(metrics["case3"]["svm"]),
            np.asarray(metrics["case4"]["svm"]),
            np.asarray(metrics["case5"]["svm"]),
        ]
    )
    worst_vm = np.max(vm_arb, axis=0)
    iar = int(np.argmin(worst_vm))
    best_arb = {"theta_deg": float(angles[iar]), "sigma_vm_pa": float(worst_vm[iar])}

    # Optional combined arbitrary-traction visualization.
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(angles, vm_arb[0], marker="o", lw=1.2, label="Case 2 (30 deg)")
    ax.plot(angles, vm_arb[1], marker="s", lw=1.2, label="Case 3 (45 deg)")
    ax.plot(angles, vm_arb[2], marker="^", lw=1.2, label="Case 4 (60 deg)")
    ax.plot(angles, vm_arb[3], marker="d", lw=1.2, label="Case 5 (+y)")
    ax.plot(angles, worst_vm, "k--", lw=2.0, label="Worst of Cases 2-5")
    ax.set_xlabel("Fiber orientation [deg]")
    ax.set_ylabel(r"$\max \sigma_{vm}$ [Pa]")
    ax.set_title("Arbitrary Right-Traction Cases: Orientation Robustness")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(frameon=False)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.savefig(out_dir / "cases2to5_orientation_robustness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    angle_rows = np.column_stack([angles, vm_arb.T, worst_vm])
    write_problem3_summary(out_dir, orth, best_case1, best_arb, angle_rows)

    print("\nRecommended fiber orientation from sweep:")
    print(
        f"  Case 1 (+x traction): theta = {best_case1['theta_deg']:.1f} deg, "
        f"min max sigma_vm = {best_case1['sigma_vm_pa']:.6e} Pa"
    )
    print(
        f"  Arbitrary right-edge tractions (Cases 2-5, minimax): theta = {best_arb['theta_deg']:.1f} deg, "
        f"worst-case sigma_vm = {best_arb['sigma_vm_pa']:.6e} Pa"
    )


if __name__ == "__main__":
    main()
