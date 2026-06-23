"""Reproduce the RIGHT panel of Figure 3 in Singh et al. JHEP 01 (2025) 098.

The panel shows the precession-angle uncertainty sigma_omega as a function of the
zonal-harmonic truncation N (how many J_l are kept in the covariance contraction),
for the conservative first-orbit geometry i = 1.57, omega = 3.08. sigma_omega rises
from ~0.81e-9 at N=2 and saturates near ~0.935e-9 by N~12, staying flat out to N=30.

    sigma_omega(N)^2 = grad_N . C_N . grad_N,
    grad_N = [ d<dw>/dmu, d<dw>/dJ2, ..., d<dw>/dJ_N ],
    C_N    = Durante covariance block for [GM, J2, ..., J_N].

The analytic Appendix-B derivatives (precession_gravity.domega_dJl) only exist for
l = 2..10. For all harmonics l = 2..30 we instead use the Gauss planetary-equations
model translated from the Mathematica notebook J_calc.nb
(precession_zonal_gauss.domega_dJl_gauss), which reproduces B.1-B.9 to ~1e-14.

Run:    python make_figure3_right.py
Output: figure3_right.png  (+ printed sigma_omega(N) table)
"""

import numpy as np
import matplotlib.pyplot as plt

import constants as C
from covariance import covariance_slice
from precession_gravity import domega_dmu
from precession_zonal_gauss import domega_dJl_gauss


def gradient_vector(a, e, inc, omega, RJ, n_max):
    """[d<dw>/dmu, d<dw>/dJ2, ..., d<dw>/dJ_{n_max}] in the paper (unnormalised) basis.

    Zonal derivatives use the J_calc.nb Gauss-equation model for all l = 2..n_max.
    """
    grad = [domega_dmu(a, e)]
    for l in range(2, n_max + 1):
        grad.append(domega_dJl_gauss(l, a, e, inc, omega, RJ))
    return np.array(grad)


def sigma_omega_N(n_max, a=C.A_KM, e=C.ECC, inc=C.INC, omega=C.OMEGA, RJ=C.R_J_KM):
    grad = gradient_vector(a, e, inc, omega, RJ, n_max)
    cov = covariance_slice(n_max)
    return np.sqrt(grad @ cov @ grad)


def main():
    N = np.arange(2, 31)
    sig = np.array([sigma_omega_N(n) for n in N])

    print(f"conservative orbit  i = {C.INC}, omega = {C.OMEGA}")
    print(f"{'N':>3} {'sigma_omega':>14}")
    for n, s in zip(N, sig):
        print(f"{n:>3} {s:14.4e}")
    print(f"\nplateau (N>=12) sigma_omega = {sig[N >= 12].mean():.4e}  "
          f"(paper Fig. 3 right: ~0.935e-9)")

    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    ax.plot(N, sig / 1e-9, "o", color="#1f77b4", ms=7)

    ax.set_xlim(0, 31)
    ax.set_ylim(0.70, 1.00)
    ax.set_xlabel(r"$N$", fontsize=13)
    ax.set_ylabel(r"$\sigma_\omega$", fontsize=13)
    ax.text(0.5, 0.93, rf"$i = {C.INC},\ \omega = {C.OMEGA}$",
            transform=ax.transAxes, ha="center", fontsize=13)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_visible(False)
    ax.text(0.0, 1.01, r"$\times 10^{-9}$", transform=ax.transAxes, fontsize=11)

    fig.tight_layout()
    out = "figure3_right.png"
    fig.savefig(out, dpi=160)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
