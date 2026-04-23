import numpy as np

# Top-panel dimensions from Problem 1 geometry
FT_TO_M = 0.3048
IN_TO_M = 0.0254

L_FT = 56.0 / 12.0
W_FT = 4.0 / 12.0
T_FT = 0.5 / 12.0

L_M = L_FT * FT_TO_M
W_M = W_FT * FT_TO_M
T_M = T_FT * FT_TO_M

E_AL = 70e9
NU_AL = 0.3


def get_panel_dimensions():
    """Return panel dimensions in SI units."""
    return {"L": L_M, "W": W_M, "t": T_M}


def generate_rect_quad_mesh(L, W, nx, ny):
    """
    Structured Quad4 mesh on a rectangle centered at origin.
    Node ordering in each element is CCW:
      n1(bottom-left), n2(bottom-right), n3(top-right), n4(top-left)
    """
    xs = np.linspace(-0.5 * L, 0.5 * L, nx + 1)
    ys = np.linspace(-0.5 * W, 0.5 * W, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    xy = np.column_stack([X.ravel(), Y.ravel()])

    conn = np.zeros((nx * ny, 4), dtype=int)
    eid = 0
    for j in range(ny):
        for i in range(nx):
            n1 = j * (nx + 1) + i
            n2 = n1 + 1
            n4 = (j + 1) * (nx + 1) + i
            n3 = n4 + 1
            conn[eid, :] = [n1, n2, n3, n4]
            eid += 1
    return xy, conn
