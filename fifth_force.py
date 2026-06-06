"""Fifth-force-induced precession drift of Juno (Singh et al. eqs. 2.3-2.5).

A Yukawa fifth force with strength alpha and range lambda gives, between Juno (point
mass m) and a Jupiter mass element, the potential of eq. (2.1)/(2.3). Treating it as a
radial perturbation, the cycle-averaged drift of the argument of perijove is (eqs. 2.4-2.5)

    <Delta omega> = -1/(e mu) * integral_0^{2pi} a_r(r) r^2 cos f df,

with mu = G M_Jupiter, r(f) = a(1 - e^2)/(1 + e cos f), and a_r the radial perturbing
acceleration. <Delta omega> is linear in alpha, so we return g(lambda) = <Delta omega>/alpha.

CONVENTION: this is the drift PER ORBIT. The Appendix-B gravity expressions used for
sigma_omega are likewise per-orbit (eq. B.1 reproduces the textbook per-revolution J2
apsidal precession 3*pi*J2*(R/p)^2*(2 - 2.5 sin^2 i), with no 1/(2 pi) factor), so both
sides of the constraint |<Delta omega>| <= 2 sigma_omega share the same per-orbit
normalisation and the 1/(2 pi) written in eq. (2.5) cancels out of the bound.

Finite-size Jupiter (eq. 2.3): for a spherically symmetric density rho(r'), the Yukawa
volume integral collapses to a 1-D radial integral. For a field point at r >= R_J (Juno
is always outside Jupiter, r_p ~ 1.06 R_J),

    U(r) = -(4 pi alpha G m lambda / r) e^{-r/lambda} * S(lambda),
    S(lambda) = integral_0^{R_J} rho(r') r' sinh(r'/lambda) dr'.

To avoid sinh/exp overflow at small lambda we fold the e^{-r/lambda} factor into S:
    sinh(r'/lambda) e^{-r/lambda} = 0.5 [ e^{(r'-r)/lambda} - e^{-(r'+r)/lambda} ],  (both <= 0 exponents).

The point-mass limit (S -> M_J for lambda >> R_J) is recovered automatically and is
also provided explicitly for cross-checks. Juno's mass cancels.
"""

import numpy as np

import constants as C
from density_profile import get_profile


def g_pointmass(lam_m, a_m=C.A_M, e=C.ECC, n_f=2000):
    """g(lambda) = <Delta omega>/alpha for a point-mass Jupiter.

    lam_m may be a scalar or array (m). Returns array matching lam_m.
    """
    lam = np.atleast_1d(np.asarray(lam_m, dtype=float))
    f = np.linspace(0.0, 2.0 * np.pi, n_f)
    r = a_m * (1.0 - e**2) / (1.0 + e * np.cos(f))            # (n_f,)
    cosf = np.cos(f)
    out = np.empty_like(lam)
    for k, L in enumerate(lam):
        integrand = np.exp(-r / L) * (1.0 + r / L) * cosf
        integral = np.trapezoid(integrand, f)
        out[k] = -integral / e            # per-orbit drift
    return out if out.size > 1 else out[0]


def g_finite_size(lam_m, a_m=C.A_M, e=C.ECC, model="polytrope",
                  n_f=1440, n_r=1500):
    """g(lambda) = <Delta omega>/alpha including Jupiter's finite size (eq. 2.3).

    Uses the radial density profile `model` (see density_profile.py).
    """
    lam = np.atleast_1d(np.asarray(lam_m, dtype=float))
    rp, rho = get_profile(model, n_points=n_r)
    M = np.trapezoid(4.0 * np.pi * rp**2 * rho, rp)            # total mass (= M_J)

    f = np.linspace(0.0, 2.0 * np.pi, n_f)
    r = a_m * (1.0 - e**2) / (1.0 + e * np.cos(f))            # (n_f,)
    cosf = np.cos(f)

    out = np.empty_like(lam)
    rho_rp = rho * rp                                         # (n_r,) weight rho(r') r'
    for k, L in enumerate(lam):
        # inner(r, L) = integral rho(r') r' sinh(r'/L) e^{-r/L} dr'   (folded, no overflow)
        # shape (n_f, n_r)
        a_exp = (rp[None, :] - r[:, None]) / L                # <= 0
        b_exp = -(rp[None, :] + r[:, None]) / L               # <= 0
        kernel = 0.5 * (np.exp(a_exp) - np.exp(b_exp))
        inner = np.trapezoid(rho_rp[None, :] * kernel, rp, axis=1)   # (n_f,)
        integrand = cosf * (1.0 + r / L) * inner
        integral = np.trapezoid(integrand, f)
        # per-orbit: g = -(4 pi lambda)/(e M) * integral
        out[k] = -(4.0 * np.pi * L / (e * M)) * integral
    return out if out.size > 1 else out[0]


def delta_omega(alpha, lam_m, finite_size=True, **kwargs):
    """Cycle-averaged precession drift <Delta omega> for given alpha and lambda."""
    g = g_finite_size(lam_m, **kwargs) if finite_size else g_pointmass(lam_m, **kwargs)
    return alpha * g


if __name__ == "__main__":
    RJ = C.R_J_M
    # Figure 2 (left): linearity in alpha at fixed lambda.
    lam = 1.0e8
    print("Fig. 2 left -- <Delta omega> linear in alpha (lambda = 1e8 m):")
    for al in (1e-9, 1e-8, 1e-7):
        print(f"  alpha={al:.0e}: <Delta omega> = {delta_omega(al, lam):.3e}")

    # Figure 2 (right): location of the peak of |g(lambda)|.
    lam_scan = np.logspace(np.log10(3e6), np.log10(3e9), 400)
    g_fs = np.abs(g_finite_size(lam_scan))
    g_pm = np.abs(g_pointmass(lam_scan))
    lpk_fs = lam_scan[np.argmax(g_fs)]
    lpk_pm = lam_scan[np.argmax(g_pm)]
    print(f"\nPeak of |g(lambda)| (finite size) at lambda = {lpk_fs:.3e} m "
          f"= {lpk_fs/RJ:.2f} R_J")
    print(f"Peak of |g(lambda)| (point mass)  at lambda = {lpk_pm:.3e} m "
          f"= {lpk_pm/RJ:.2f} R_J")

    # Large-lambda: finite size must converge to the point-mass result.
    big = 1.0e11
    print(f"\nLarge-lambda convergence (lambda = {big:.0e} m):")
    print(f"  g_finite = {g_finite_size(big):.4e}, g_point = {g_pointmass(big):.4e}, "
          f"ratio = {g_finite_size(big)/g_pointmass(big):.4f}")
