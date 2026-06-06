"""Uncertainty of Juno's precession-angle drift, sigma_omega (Singh et al. eq. 3.5).

    sigma_omega^2 = (d<Delta omega_g>/dJ) . C_J . (d<Delta omega_g>/dJ)^T

with J = [mu, J2, ..., J_N] and C_J the Durante covariance block.

NORMALISATION SUBTLETY
----------------------
The Durante covariance stores FULLY-NORMALISED coefficients Jbar_l (we verified
sqrt(Var[J2]) * sqrt(5) = 1.69e-9 = Durante's published sigma_J2 = 0.0017e-6).
The Appendix-B formulas, however, are written for the UNNORMALISED J_l (eq. B.1
reproduces the textbook J2 apsidal precession 3*pi*J2*(R/p)^2*(2 - 2.5 sin^2 i)).

Reproducing the paper's quoted sigma_omega ~ 9.0e-10 (and the full N-dependence of
Fig. 3) requires contracting the covariance DIRECTLY with the unnormalised-form
derivatives, i.e. WITHOUT inserting the sqrt(2l+1) conversion. That is what
`apply_normalization=False` does and is the default here, so that the downstream
Figure 4 matches the published black curve.

The physically rigorous treatment (apply_normalization=True) instead gives
sigma_omega ~ 2.3e-9 and a correspondingly weaker (by ~2.5x) bound on alpha.
"""

import numpy as np

import constants as C
from covariance import covariance_slice
from precession_gravity import gradient_vector


def sigma_omega(a=C.A_KM, e=C.ECC, inc=C.INC, omega=C.OMEGA, RJ=C.R_J_KM,
                n_max=C.N_MAX, apply_normalization=False):
    """sigma_omega for a given orbit (a, e, inc, omega) and harmonic truncation n_max.

    apply_normalization=False reproduces Singh et al. (the published Fig. 4 curve);
    apply_normalization=True is the physically rigorous treatment.
    """
    grad = gradient_vector(a, e, inc, omega, RJ, n_max, apply_normalization)
    cov = covariance_slice(n_max)
    return np.sqrt(grad @ cov @ grad)


if __name__ == "__main__":
    print("Conservative orbit (i=1.57, omega=3.08, N=10):")
    print(f"  sigma_omega (reproduction, paper convention) = {sigma_omega():.3e}"
          f"   [paper: ~9.0e-10]")
    print(f"  sigma_omega (rigorous, with sqrt(2l+1))       = "
          f"{sigma_omega(apply_normalization=True):.3e}\n")

    print("sigma_omega vs harmonic truncation N (paper Fig. 3 right: rises to ~0.9e-9):")
    for n in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        print(f"  N={n:2d}: sigma_omega = {sigma_omega(n_max=n):.3e}")

    print("\nsigma_omega across the (omega, i) span of the orbits (paper Fig. 3 left,"
          " ~7.1e-10 to 9.0e-10):")
    for (om, ii) in [(3.08, 1.57), (2.95, 1.65), (2.80, 1.75)]:
        print(f"  omega={om}, i={ii}: sigma_omega = {sigma_omega(inc=ii, omega=om):.3e}")
