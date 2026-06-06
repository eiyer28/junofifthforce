"""Reproduce Figure 2 of Singh et al. JHEP 01 (2025) 098 -- the fifth-force precession drift.

Figure 2 plots the cycle-averaged (per-orbit) precession drift <Delta omega> = alpha * g(lambda)
induced by a Yukawa fifth force, where g(lambda) = <Delta omega>/alpha comes from the
finite-size calculation in fifth_force.py (eqs. 2.3-2.5). This is the same quantity that is
inverted into the bound of Figure 4; here it is plotted directly.

  Left  : <Delta omega> vs alpha (1e-9..1e-7) at fixed lambda = 1e7 m and 1e8 m.
  Right : <Delta omega> vs lambda (1e6..1e12 m) at fixed alpha = 1e-7 and 1e-6, with a
          vertical line at lambda = R_J and a secondary m* = hbar c / lambda axis.

Run:  python make_figure2.py
Output: figure2_deltaomega.png
"""

import numpy as np
import matplotlib.pyplot as plt

import constants as C
from fifth_force import g_finite_size


def _m_star_transforms():
    """Forward/inverse transforms between lambda [m] and mediator mass m* [eV]."""
    def lam_to_m(l):
        l = np.asarray(l, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return C.HBAR_EV_S * C.C_M_S / l

    def m_to_lam(m):
        m = np.asarray(m, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return C.HBAR_EV_S * C.C_M_S / m

    return lam_to_m, m_to_lam


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # --- Left panel: <Delta omega> vs alpha at fixed lambda -----------------
    alpha = np.logspace(-9, -7, 50)
    for lam0, style in [(1e7, "C0-"), (1e8, "C1--")]:
        g0 = abs(g_finite_size(lam0))                 # scalar; <Delta omega> = alpha * g0
        axL.loglog(alpha, alpha * g0, style, lw=2.0,
                   label=rf"$\lambda = 10^{{{int(np.log10(lam0))}}}$ m")
    axL.set_xlim(1e-9, 1e-7)
    axL.set_ylim(1e-10, 1e-6)
    axL.set_xlabel(r"$\alpha$")
    axL.set_ylabel(r"$\Delta\omega$")
    axL.grid(True, which="both", ls=":", alpha=0.4)
    axL.legend(loc="lower right", frameon=True)

    # --- Right panel: <Delta omega> vs lambda at fixed alpha ----------------
    lam = np.logspace(6, 12, 160)
    g = np.abs(g_finite_size(lam))                    # reused for both alpha curves
    for alpha0, style in [(1e-7, "C0-"), (1e-6, "C1--")]:
        axR.loglog(lam, alpha0 * g, style, lw=2.0,
                   label=rf"$\alpha = 10^{{{int(np.log10(alpha0))}}}$")
    axR.axvline(C.R_J_M, ls="--", color="0.6", lw=1.2)
    axR.text(C.R_J_M * 1.1, 2e-13, r"$\lambda = R_J$", color="0.4", fontsize=9)
    axR.set_xlim(1e6, 1e12)
    axR.set_ylim(1e-13, 1e-6)
    axR.set_xlabel(r"$\lambda$ [m]")
    axR.set_ylabel(r"$\Delta\omega$")
    axR.grid(True, which="both", ls=":", alpha=0.4)
    axR.legend(loc="lower center", frameon=True)

    lam_to_m, m_to_lam = _m_star_transforms()
    secax = axR.secondary_xaxis("top", functions=(lam_to_m, m_to_lam))
    secax.set_xlabel(r"$m_*$ [eV]")

    fig.suptitle("Fifth-force precession drift (reproduction of Fig. 2, JHEP 01 (2025) 098)")
    fig.tight_layout()
    out = "figure2_deltaomega.png"
    fig.savefig(out, dpi=160)

    # --- Validation printout ----------------------------------------------
    kpk = np.argmax(g)
    print(f"peak of |g(lambda)| at lambda = {lam[kpk]:.3e} m = {lam[kpk]/C.R_J_M:.2f} R_J")
    g1e8 = abs(g_finite_size(1e8))
    print(f"linearity check (lambda=1e8 m): "
          f"<Delta omega>(alpha=1e-9) = {1e-9 * g1e8:.3e}, "
          f"<Delta omega>(alpha=1e-8) = {1e-8 * g1e8:.3e}  (ratio 10)")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
