import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter


def summarize_case(name, out):
    sig = out["element_stress"]
    vm = out["element_vm"]
    print(f"\n{name}")
    print(f"  max nodal displacement magnitude: {out['max_u']:.6e} m")
    print(f"  sigma_x range: [{np.min(sig[:,0]):.6e}, {np.max(sig[:,0]):.6e}] Pa")
    print(f"  sigma_y range: [{np.min(sig[:,1]):.6e}, {np.max(sig[:,1]):.6e}] Pa")
    print(f"  tau_xy  range: [{np.min(sig[:,2]):.6e}, {np.max(sig[:,2]):.6e}] Pa")
    print(f"  max von Mises: {np.max(vm):.6e} Pa")


def displacement_scale(xy, U_nodes, factor=0.15):
    max_u = np.max(np.linalg.norm(U_nodes, axis=1))
    if max_u <= 0:
        return 1.0
    Lc = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
    Lc = max(float(Lc), 1e-12)
    return factor * Lc / max_u


def _style_axes(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]", labelpad=6)
    ax.set_ylabel("y [m]")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_mesh_lines(ax, xy, conn, color="k", lw=0.8, alpha=1.0):
    for e in range(conn.shape[0]):
        n = conn[e]
        p = xy[[n[0], n[1], n[2], n[3], n[0]], :]
        ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, alpha=alpha)


def _triangulation_from_quads(xy, conn):
    tris = np.zeros((2 * conn.shape[0], 3), dtype=int)
    for e, n in enumerate(conn):
        tris[2 * e] = [n[0], n[1], n[2]]
        tris[2 * e + 1] = [n[0], n[2], n[3]]
    return mtri.Triangulation(xy[:, 0], xy[:, 1], tris)


def _edge_node_ids(xy, edge, tol=1e-10):
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
    return nodes


def _edge_corners(xy, edge, tol=1e-10):
    nodes = _edge_node_ids(xy, edge, tol=tol)
    pts = xy[nodes]
    if edge in ("left", "right"):
        order = np.argsort(pts[:, 1])
    else:
        order = np.argsort(pts[:, 0])
    p0 = pts[order[0]]
    p1 = pts[order[-1]]
    return p0, p1


def _edge_length(xy, edge, tol=1e-10):
    p0, p1 = _edge_corners(xy, edge, tol=tol)
    return float(np.linalg.norm(p1 - p0))


def _constrained_dofs_by_edge(xy, bcs, tol=1e-10):
    # Keep dof only if every node on that edge is constrained in that dof.
    edge_info = {}
    for edge in ("left", "right", "bottom", "top"):
        edge_nodes = _edge_node_ids(xy, edge, tol=tol)
        edge_nodes_set = set(int(n) for n in edge_nodes)
        if not edge_nodes_set:
            continue

        node_to_dofs = {}
        for bc in bcs:
            if bc.get("type", "dirichlet") != "dirichlet":
                continue
            n = int(bc["node"])
            if n in edge_nodes_set:
                node_to_dofs.setdefault(n, set()).add(str(bc["dof"]))

        dofs_all = []
        for dof in ("ux", "uy"):
            if all((n in node_to_dofs and dof in node_to_dofs[n]) for n in edge_nodes_set):
                dofs_all.append(dof)
        if dofs_all:
            edge_info[edge] = tuple(dofs_all)
    return edge_info


def _format_dof_tex(dofs):
    comps = []
    if "ux" in dofs:
        comps.append("x")
    if "uy" in dofs:
        comps.append("y")
    if len(comps) == 1:
        return rf"$u_{{{comps[0]}}}=0$"
    return rf"$u_{{{','.join(comps)}}}=0$"


def _plot_constraint_edges(ax, xy, bcs):
    edge_info = _constrained_dofs_by_edge(xy, bcs)
    handles = []
    labels = []
    for edge, dofs in edge_info.items():
        p0, p1 = _edge_corners(xy, edge)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="tab:blue", lw=2.2, zorder=4)
        handles.append(Line2D([0], [0], color="tab:blue", lw=2.2))
        labels.append(f"BC: {_format_dof_tex(dofs)}")
    # De-duplicate repeated BC labels while preserving order.
    uniq = []
    for h, l in zip(handles, labels):
        if l not in [x[1] for x in uniq]:
            uniq.append((h, l))
    if len(uniq) > 0:
        handles, labels = zip(*uniq)
        return list(handles), list(labels)
    return handles, labels


def _plot_traction_vectors(ax, xy, loads):
    if not loads:
        return [], []
    span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-12)
    arrow_len = 0.05 * span
    handles = []
    labels = []

    for ld in loads:
        if ld.get("type") != "traction_edge":
            continue
        edge = ld["edge"]
        tvec = np.asarray(ld["value"], dtype=float)
        mag = float(np.linalg.norm(tvec))
        if mag <= 0.0:
            continue

        p0, p1 = _edge_corners(xy, edge)
        tdir = tvec / mag
        dx, dy = arrow_len * tdir[0], arrow_len * tdir[1]

        edge_len = _edge_length(xy, edge)
        corner_force = 0.5 * mag * edge_len  # N per corner node

        for p in (p0, p1):
            ax.quiver(
                p[0],
                p[1],
                dx,
                dy,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="tab:red",
                width=0.003,
                zorder=5,
                clip_on=False,
            )
        dominant = tvec[1] if abs(tvec[1]) >= abs(tvec[0]) else tvec[0]
        corner_force_signed = np.sign(dominant) * corner_force
        handles.append(Line2D([0], [0], color="tab:red", lw=2.0))
        labels.append(f"traction: {corner_force_signed:.2f} N")
    return handles, labels


def plot_undeformed_mesh(xy, conn, case, out_path):
    fig, ax = plt.subplots(figsize=(8, 3.0))
    _plot_mesh_lines(ax, xy, conn, color="k", lw=0.8)
    bc_handles, bc_labels = _plot_constraint_edges(ax, xy, case.get("bcs", []))
    tr_handles, tr_labels = _plot_traction_vectors(ax, xy, case.get("loads", []))
    handles = bc_handles + tr_handles
    labels = bc_labels + tr_labels
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper left",
            fontsize=8,
            frameon=False,
        )

    xmin, xmax = float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))
    dx = max(xmax - xmin, 1e-12)
    ax.set_xlim(xmin - 0.04 * dx, xmax + 0.04 * dx)
    ax.set_ylim(-0.2, 0.2)
    _style_axes(ax)
    ax.set_title(f"{case['name']}: Undeformed Mesh")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_deformed_mesh(xy, conn, U_nodes, title, out_path, scale=None):
    y_min_lim, y_max_lim = -0.2, 0.2
    if scale is None:
        scale = displacement_scale(xy, U_nodes)
        # Cap scale so deformed y remains within fixed plot limits.
        y0 = xy[:, 1]
        uy = U_nodes[:, 1]
        bounds = []
        pos = uy > 1e-16
        neg = uy < -1e-16
        if np.any(pos):
            bounds.extend(((y_max_lim - y0[pos]) / uy[pos]).tolist())
        if np.any(neg):
            bounds.extend(((y_min_lim - y0[neg]) / uy[neg]).tolist())
        bounds = [b for b in bounds if b > 0.0 and np.isfinite(b)]
        if len(bounds) > 0:
            scale = min(scale, 0.95 * min(bounds))
    xy_def = xy + scale * U_nodes

    fig, ax = plt.subplots(figsize=(8, 3.0))
    _plot_mesh_lines(ax, xy, conn, color="0.70", lw=0.7)
    _plot_mesh_lines(ax, xy_def, conn, color="tab:red", lw=0.9)
    x_all = np.concatenate([xy[:, 0], xy_def[:, 0]])
    xmin, xmax = float(np.min(x_all)), float(np.max(x_all))
    dx = max(xmax - xmin, 1e-12)
    ax.set_xlim(xmin - 0.04 * dx, xmax + 0.04 * dx)
    ax.set_ylim(y_min_lim, y_max_lim)
    handles = [
        Line2D([0], [0], color="0.70", lw=1.2, label="undeformed"),
        Line2D([0], [0], color="tab:red", lw=1.2, label="deformed"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)
    _style_axes(ax)
    ax.set_title(f"{title}: Deformed Mesh (scale={scale:.2e})")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return scale


def plot_stress_contour(xy, conn, nodal_scalar_pa, label, title, out_path, cmap="turbo"):
    triang = _triangulation_from_quads(xy, conn)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    tcf = ax.tricontourf(triang, nodal_scalar_pa, levels=20, cmap=cmap)
    ax.triplot(triang, color="k", lw=0.25, alpha=0.45)

    cbar = plt.colorbar(tcf, ax=ax, orientation="horizontal", pad=0.22, fraction=0.08)
    cbar.set_label(label)
    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2e"))
    cbar.update_ticks()

    _style_axes(ax)
    ax.set_title(title)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_stress_stack(
    xy,
    conn,
    nodal_sx,
    nodal_sy,
    nodal_vm,
    case_title,
    out_path,
    cmap="turbo",
    shared_scale=False,
):
    triang = _triangulation_from_quads(xy, conn)
    fields = [np.asarray(nodal_sx), np.asarray(nodal_sy), np.asarray(nodal_vm)]
    subtitles = [r"$\sigma_x$", r"$\sigma_y$", r"$\sigma_{vm}$"]

    if shared_scale:
        vmin = min(float(np.min(f)) for f in fields)
        vmax = max(float(np.max(f)) for f in fields)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        shared_levels = np.linspace(vmin, vmax, 25)
    else:
        shared_levels = None

    fig, axes = plt.subplots(3, 1, figsize=(8, 7.0))
    fig.subplots_adjust(left=0.09, right=0.94, top=0.92, bottom=0.08, hspace=0.65)
    for ax, data, subtitle in zip(axes, fields, subtitles):
        if shared_scale:
            contour_ref = ax.tricontourf(
                triang, data, levels=shared_levels, cmap=cmap, vmin=vmin, vmax=vmax
            )
        else:
            contour_ref = ax.tricontourf(triang, data, levels=20, cmap=cmap)
        ax.triplot(triang, color="k", lw=0.20, alpha=0.35)
        _style_axes(ax)
        ax.set_title(subtitle)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="22%", pad=0.55)
        cbar = fig.colorbar(contour_ref, cax=cax, orientation="horizontal")
        cbar.set_label("Pa")
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2e"))
        cbar.update_ticks()

    fig.suptitle(case_title, fontsize=12)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_case_outputs(out_dir, case_tag, xy, conn, out):
    nnode = xy.shape[0]
    node_ids = np.arange(nnode, dtype=int)
    node_table = np.column_stack([node_ids, xy, out["U_nodes"], out["u_mag"]])
    np.savetxt(
        out_dir / f"{case_tag}_node_displacements.csv",
        node_table,
        delimiter=",",
        header="node,x_m,y_m,ux_m,uy_m,u_mag_m",
        comments="",
        fmt=["%d", "%.8e", "%.8e", "%.8e", "%.8e", "%.8e"],
    )

    eids = np.arange(conn.shape[0], dtype=int)
    est = out["element_stress"]
    etable = np.column_stack([eids, conn, est, out["element_vm"]])
    np.savetxt(
        out_dir / f"{case_tag}_element_stress.csv",
        etable,
        delimiter=",",
        header="element,n1,n2,n3,n4,sigma_x_pa,sigma_y_pa,tau_xy_pa,sigma_vm_pa",
        comments="",
        fmt=["%d", "%d", "%d", "%d", "%d", "%.8e", "%.8e", "%.8e", "%.8e"],
    )

    with open(out_dir / f"{case_tag}_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"max_u_m={out['max_u']:.8e}\n")
        f.write(f"max_abs_sigma_x_pa={np.max(np.abs(out['element_stress'][:, 0])):.8e}\n")
        f.write(f"max_abs_sigma_y_pa={np.max(np.abs(out['element_stress'][:, 1])):.8e}\n")
        f.write(f"max_sigma_vm_pa={np.max(out['element_vm']):.8e}\n")


def save_case_plots(out_dir, case_tag, case, out):
    xy = case["mesh"]["xy"]
    conn = case["mesh"]["conn"]
    case_title = case["name"]

    plot_undeformed_mesh(
        xy,
        conn,
        case=case,
        out_path=out_dir / f"{case_tag}_undeformed.png",
    )
    scale = plot_deformed_mesh(
        xy,
        conn,
        out["U_nodes"],
        title=case_title,
        out_path=out_dir / f"{case_tag}_deformed.png",
    )
    plot_stress_stack(
        xy,
        conn,
        out["nodal_stress"][:, 0],
        out["nodal_stress"][:, 1],
        out["nodal_vm"],
        case_title=case_title,
        out_path=out_dir / f"{case_tag}_stress_stack.png",
        shared_scale=False,
    )
    return scale


def plot_case_convergence(out_dir, case_tag, case_title, elem_counts, sx_vals, sy_vals, svm_vals):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(elem_counts, sx_vals, marker="o", lw=1.6, label=r"$\max|\sigma_x|$")
    ax.plot(elem_counts, sy_vals, marker="s", lw=1.6, label=r"$\max|\sigma_y|$")
    ax.plot(elem_counts, svm_vals, marker="^", lw=1.6, label=r"$\max\sigma_{vm}$")

    ax.set_xlabel("Number of elements")
    ax.set_ylabel("Stress [Pa]")
    ax.set_title(f"{case_title}: Stress Convergence")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(frameon=False)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.savefig(out_dir / f"{case_tag}_convergence_stress.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_orientation_sweep(out_dir, case_tag, case_title, angles_deg, sx_vals, sy_vals, svm_vals):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(angles_deg, sx_vals, marker="o", lw=1.6, label=r"$\max|\sigma_x|$")
    ax.plot(angles_deg, sy_vals, marker="s", lw=1.6, label=r"$\max|\sigma_y|$")
    ax.plot(angles_deg, svm_vals, marker="^", lw=1.6, label=r"$\max\sigma_{vm}$")

    ax.set_xlabel("Fiber orientation [deg]")
    ax.set_ylabel("Stress [Pa]")
    ax.set_title(f"{case_title}: Stress vs Fiber Orientation")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(frameon=False)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.savefig(out_dir / f"{case_tag}_orientation_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
