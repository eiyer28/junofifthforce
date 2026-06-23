"""General-order zonal-harmonic precession derivative d<Delta omega_g>/dJ_l,
translated from the Mathematica notebook J_calc.nb (Gauss planetary-equations model).

This replaces the earlier Lagrange disturbing-function implementation. The two are
mathematically equivalent (both reproduce Appendix-B B.1-B.9 of Singh et al.), but
this one follows the notebook's logic exactly.

NOTEBOOK RECIPE
---------------
1. Potential of zonal harmonic m (per unit mass):
       U_m = -(mu/r) (RJ/r)^m P_m(z/r) J_m.
2. Force = grad(U_m) in cylindrical coords; projected to radial/transverse/normal
   (R-T-N) with the substitution z -> r sin(i) sin(theta), theta = argument of
   latitude. With s = sin(i) sin(theta) and the prefactor K = (1+m)(RJ/r)^m (mu/r^2):
       F_R =  K P_m(s) J_m
       F_T = -K sin(i) cos(theta) P_m'(s)/(1+m) J_m
       F_N = -K cos(i)            P_m'(s)/(1+m) J_m
   (The notebook writes these with P_{m+1}; using the Legendre recurrence
    (1-x^2)P_m'(x) = (m+1)(x P_m - P_{m+1}), the factor (s P_m - P_{m+1})/(s^2-1)
    is exactly -P_m'(s)/(m+1), which is regular at the poles s -> +/-1.)
3. Gauss equations, secularly averaged over one orbit (theta: 0..2pi, f = theta - omega,
   r = a(1-e^2)/(1+e cos f), n = sqrt(mu/a^3)):
       Omega_m = (n/2pi) integral  F_N sin(theta)/(sin(i)(1+e cos f)) (r^2/mu) dtheta
       omega_m = -cos(i) Omega_m
                 + (n/2pi) integral [ -F_R cos f
                                      + F_T (2+e cos f) sin f/(1+e cos f) ] (r^2/(e mu)) dtheta
   omega_m is the time-averaged drift RATE of the argument of perijove.
4. Per-orbit drift (paper convention, eqs. B.1-B.9):
       d<Delta omega_g>/dJ_m = (2pi/n) omega_m.
   The n and mu factors cancel, leaving a dimensionless quantity (lengths in km).
"""

import numpy as np
from numpy.polynomial import legendre as L

import constants as C


def _legP(m, x):
    """Legendre polynomial P_m(x) (exact, via the Legendre basis)."""
    c = np.zeros(m + 1)
    c[m] = 1.0
    return L.legval(x, c)


def _legP_deriv(m, x):
    """Derivative P_m'(x) (exact polynomial differentiation, regular at x=+/-1)."""
    c = np.zeros(m + 1)
    c[m] = 1.0
    return L.legval(x, L.legder(c))


def domega_dJl_gauss(m, a=C.A_KM, e=C.ECC, inc=C.INC, omega=C.OMEGA,
                     RJ=C.R_J_KM, n_f=20000):
    """d<Delta omega_g>/dJ_m for zonal order m>=2 (per orbit), via the notebook's
    Gauss-equation model. Lengths in km; result dimensionless (paper convention,
    matching precession_gravity.domega_dJl).
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_f)
    f = theta - omega
    cos_f = np.cos(f)
    sin_f = np.sin(f)
    one_ecf = 1.0 + e * cos_f
    r = a * (1.0 - e**2) / one_ecf

    s = np.sin(inc) * np.sin(theta)              # = z/r on the orbit
    Pm = _legP(m, s)
    dPm = _legP_deriv(m, s)
    RJr_m = (RJ / r) ** m

    # Force components with the r^2/mu factor already folded in (mu, (1+m) cancel):
    FR_r2_mu = (1.0 + m) * RJr_m * Pm                          # F_R r^2/mu
    FT_r2_mu = -RJr_m * np.sin(inc) * np.cos(theta) * dPm      # F_T r^2/mu
    FN_r2_mu = -RJr_m * np.cos(inc) * dPm                      # F_N r^2/mu

    # Integrands (per unit J_m).  d<Delta omega>/dJ_m = (2pi/n) omega_m, and the
    # (2pi/n)(n/2pi) prefactors cancel, so we integrate the bracketed terms directly.
    term_omega = (-FR_r2_mu * cos_f
                  + FT_r2_mu * (2.0 + e * cos_f) * sin_f / one_ecf) / e
    term_Omega = FN_r2_mu * np.sin(theta) / (np.sin(inc) * one_ecf)

    integrand = term_omega - np.cos(inc) * term_Omega
    return np.trapezoid(integrand, theta)


# Backwards-compatible alias so existing imports keep working.
domega_dJl_general = domega_dJl_gauss


def validate(a=C.A_KM, e=C.ECC, inc=C.INC, omega=C.OMEGA, RJ=C.R_J_KM):
    """Compare the notebook (Gauss) derivative with analytic B.1-B.9 (l=2..10)."""
    from precession_gravity import domega_dJl
    print(f"{'l':>3} {'analytic (B.l)':>16} {'gauss (J_calc)':>16} {'rel.err':>10}")
    for l in range(2, 11):
        ana = domega_dJl(l, a, e, inc, omega, RJ)
        num = domega_dJl_gauss(l, a, e, inc, omega, RJ)
        rel = abs(num - ana) / abs(ana)
        print(f"{l:>3} {ana:16.6e} {num:16.6e} {rel:10.2e}")


if __name__ == "__main__":
    validate()
