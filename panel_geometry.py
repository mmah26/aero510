import numpy as np

# Input dimensions provided in inches, converted to SI (meters).
IN_TO_M = 0.0254
L_IN = 56.0
W_IN = 4.0
T_IN = 0.5

L_M = L_IN * IN_TO_M
W_M = W_IN * IN_TO_M
T_M = T_IN * IN_TO_M

# Material properties (SI)
E_AL = 70.0e9  # Pa
NU_AL = 0.3


def gen_quad_mesh(L, W, nx, ny):
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
