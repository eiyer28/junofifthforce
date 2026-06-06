"""Reproduce the black solid Juno curve of Figure 4 in Singh et al. JHEP 01 (2025) 098.

For each fifth-force range lambda we exclude (at 95% C.L.) all strengths |alpha| whose
induced per-orbit precession drift |<Delta omega>(alpha, lambda)| exceeds 2 sigma_omega,
i.e. the limiting strength is

    alpha_limit(lambda) = 2 * sigma_omega / |g(lambda)|,   g(lambda) = <Delta omega>/alpha.

sigma_omega (eq. 3.5) comes from the Durante gravity covariance; g(lambda) (eqs. 2.3-2.5)
is the Yukawa precession including Jupiter's finite size. The region ABOVE the curve is
ruled out.

Run:  python make_figure4.py
Output: figure4_juno.png  (plus printed validation numbers)
"""

import numpy as np
import matplotlib.pyplot as plt

import constants as C
from sigma_omega import sigma_omega
from fifth_force import g_finite_size


def juno_bound(lam_m, n_sigma=2.0, density_model="polytrope",
               apply_normalization=False):
    """alpha_limit(lambda) for the Juno fifth-force constraint."""
    s = sigma_omega(apply_normalization=apply_normalization)
    g = g_finite_size(lam_m, model=density_model)
    return n_sigma * s / np.abs(g), s


def main():
    lam = np.logspace(5, 12, 160)          # 1e5 .. 1e12 m
    alpha_limit, s = juno_bound(lam)

    kmin = np.argmin(alpha_limit)
    lam_opt, a_opt = lam[kmin], alpha_limit[kmin]

    print(f"sigma_omega                 = {s:.3e}  (paper ~9.0e-10)")
    print(f"strongest bound  alpha_min  = {a_opt:.3e}  (paper ~1e-9)")
    print(f"   at lambda                = {lam_opt:.3e} m = {lam_opt/C.R_J_M:.2f} R_J"
          f"  (paper: optimum ~1e8 m ~ O(R_J))")

    # Asymptotic scalings (paper section 4): alpha ~ lambda^2 for lambda >> 1e8 m;
    # alpha ~ exp(R_J/lambda) (exponential weakening) for lambda << 1e7 m.
    def slope(l0, l1):
        i0 = np.argmin(np.abs(lam - l0))
        i1 = np.argmin(np.abs(lam - l1))
        return (np.log(alpha_limit[i1]) - np.log(alpha_limit[i0])) \
            / (np.log(lam[i1]) - np.log(lam[i0]))
    print(f"\nlog-log slope for lambda >> R_J (1e10->1e11 m) = "
          f"{slope(1e10, 1e11):.2f}  (expect ~ +2)")

    # --- Figure -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    ax.loglog(lam, alpha_limit, "k-", lw=2.2, label="Juno (this work)")
    ax.fill_between(lam, alpha_limit, 1e3, color="0.85", alpha=0.5, zorder=0)

    ax.set_xlim(1e5, 1e12)
    ax.set_ylim(1e-11, 1e-3)
    ax.set_xlabel(r"$\lambda$ [m]")
    ax.set_ylabel(r"$|\alpha|$")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # Marker at the optimum.
    ax.plot([lam_opt], [a_opt], "o", color="crimson", ms=6, zorder=5)
    ax.annotate(rf"$\alpha_{{\min}}\approx{a_opt:.1e}$",
                xy=(lam_opt, a_opt), xytext=(lam_opt * 3, a_opt * 0.25),
                fontsize=9, color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8))

    # Secondary top axis: mediator mass m* = hbar c / lambda.
    def lam_to_m(l):
        l = np.asarray(l, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return C.HBAR_EV_S * C.C_M_S / l

    def m_to_lam(m):
        m = np.asarray(m, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return C.HBAR_EV_S * C.C_M_S / m

    secax = ax.secondary_xaxis("top", functions=(lam_to_m, m_to_lam))
    secax.set_xlabel(r"$m_*$ [eV]")

    ax.legend(loc="lower right", frameon=True)
    ax.set_title("Juno fifth-force constraint (reproduction of Fig. 4, JHEP 01 (2025) 098)")
    fig.tight_layout()
    out = "figure4_juno.png"
    fig.savefig(out, dpi=160)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
