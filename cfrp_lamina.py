import numpy as np


def shear_modulus(E, nu):
    return E / (2.0 * (1.0 + nu))


def halpin_tsai(Pf, Pm, Vf, xi):
    """
    Halpin-Tsai estimate for a lamina property transverse to fiber direction.
    Pf: fiber property, Pm: matrix property, Vf: fiber volume fraction, xi: shape parameter.
    """
    if Pm <= 0.0:
        raise ValueError("Matrix property must be positive for Halpin-Tsai.")
    r = Pf / Pm
    eta = (r - 1.0) / (r + xi)
    return Pm * (1.0 + xi * eta * Vf) / (1.0 - eta * Vf)


def cfrp_ud_effective_properties(
    Ef,
    nuf,
    Em,
    num,
    Vf,
    xi_E2=2.0,
    xi_G12=1.0,
):
    """
    Unidirectional CFRP effective in-plane orthotropic constants.
    Inputs in SI units (Pa, unitless Poisson ratio).
    Returns E1, E2, G12, nu12, and auxiliary terms.
    """
    Vm = 1.0 - Vf
    if Vf <= 0.0 or Vf >= 1.0:
        raise ValueError("Vf must be between 0 and 1.")

    Gf = shear_modulus(Ef, nuf)
    Gm = shear_modulus(Em, num)

    # Longitudinal rule of mixtures.
    E1 = Vf * Ef + Vm * Em
    # Transverse and in-plane shear via Halpin-Tsai.
    E2 = halpin_tsai(Ef, Em, Vf, xi=xi_E2)
    G12 = halpin_tsai(Gf, Gm, Vf, xi=xi_G12)
    # Major Poisson's ratio by mixture rule.
    nu12 = Vf * nuf + Vm * num

    return {
        "E1": float(E1),
        "E2": float(E2),
        "G12": float(G12),
        "nu12": float(nu12),
        "Gf": float(Gf),
        "Gm": float(Gm),
    }


def traction_vector_from_angle(magnitude, angle_deg):
    th = np.deg2rad(angle_deg)
    return np.array([magnitude * np.cos(th), magnitude * np.sin(th)], dtype=float)
