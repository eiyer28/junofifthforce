"""Compare the Juno fifth-force constraint from the Durante (2020) gravity covariance
against the updated covariance of Kaspi et al. (2023).

  Durante et al., Geophys. Res. Lett. 47 (2020) e86572
  Kaspi et al., "Observational evidence for cylindrically oriented zonal flows on
    Jupiter", Nature Astronomy 7 (2023) 1463  [doi:10.1038/s41550-023-02077-8]
    (covariance provided by Y. Kaspi; PJ37 normal-modes solution)

Both covariances are full matrices in the same fully-normalised [GM, J2..J10] basis, so
the comparison is a clean slice-and-contract through identical machinery:

    sigma_omega = sqrt(grad . C . grad)              (eq. 3.5)
    alpha_limit(lambda) = 2 * sigma_omega / |g(lambda)|   (95% C.L., eq. 2.5)

Because both bounds run through the same gradient and the same g(lambda), the
old/new improvement factor is independent of any overall 1-sigma vs 3-sigma scaling.

Run:  python compare_constraints.py
Output: figure4_comparison.png  (plus printed comparison numbers)
"""

import numpy as np
import matplotlib.pyplot as plt

import constants as C
from covariance import covariance_slice, covariance_slice_new
from precession_gravity import gradient_vector
from sigma_omega import sigma_omega
from fifth_force import g_finite_size

N_SIGMA = 2.0          # 95% C.L. threshold (Singh et al.)
LABELS = ["GM"] + [f"J{l}" for l in range(2, C.N_MAX + 1)]


def alpha_curve(lam_m, cov, density_model="polytrope"):
    """alpha_limit(lambda) = N_SIGMA * sigma_omega / |g(lambda)| for a covariance block."""
    s = sigma_omega(cov=cov)
    g = g_finite_size(lam_m, model=density_model)
    return N_SIGMA * s / np.abs(g), s


def contribution_table(cov, n_max=C.N_MAX):
    """sigma_omega built up term-by-term: report each parameter's marginal addition
    (in quadrature-with-correlations sense) by truncating the block at successive sizes.
    """
    grad = gradient_vector(C.A_KM, C.ECC, C.INC, C.OMEGA, C.R_J_KM, n_max,
                           apply_normalization=False)
    cumulative = []
    for k in range(1, n_max + 1):
        g = grad[:k]
        cumulative.append(float(np.sqrt(g @ cov[:k, :k] @ g)))
    return cumulative


def main():
    cov_old = covariance_slice(C.N_MAX)
    cov_new = covariance_slice_new(C.N_MAX)

    # --- sigma_omega ------------------------------------------------------
    s_old = sigma_omega(cov=cov_old)
    s_new = sigma_omega(cov=cov_new)
    print(f"sigma_omega (Durante 2020) = {s_old:.3e}   [paper ~9.0e-10]")
    print(f"sigma_omega (Kaspi 2023)   = {s_new:.3e}")
    print(f"  -> sigma_omega improvement factor = {s_old / s_new:.2f}x\n")

    # --- alpha bound ------------------------------------------------------
    lam = np.logspace(5, 12, 200)
    a_old, _ = alpha_curve(lam, cov_old)
    a_new, _ = alpha_curve(lam, cov_new)

    i_old = int(np.argmin(a_old))
    i_new = int(np.argmin(a_new))
    print(f"strongest bound (old): alpha_min = {a_old[i_old]:.3e} at "
          f"lambda = {lam[i_old]:.3e} m  ({lam[i_old]/C.R_J_M:.2f} R_J)")
    print(f"strongest bound (new): alpha_min = {a_new[i_new]:.3e} at "
          f"lambda = {lam[i_new]:.3e} m  ({lam[i_new]/C.R_J_M:.2f} R_J)")
    print(f"  -> alpha_min improvement factor = {a_old[i_old] / a_new[i_new]:.2f}x\n")

    # --- where the improvement comes from --------------------------------
    cum_old = contribution_table(cov_old)
    cum_new = contribution_table(cov_new)
    print("cumulative sigma_omega as the block grows [GM, +J2, +J3, ...]:")
    print(f"  {'param':<6} {'old':>10} {'new':>10}")
    for lab, co, cn in zip(LABELS, cum_old, cum_new):
        print(f"  {lab:<6} {co:>10.3e} {cn:>10.3e}")
    print()

    # --- figure -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.loglog(lam, a_old, "k-", lw=2.2, label="Durante et al. 2020 (PJ01-17)")
    ax.loglog(lam, a_new, "C0-", lw=2.2, label="Kaspi et al. 2023 (PJ37)")
    ax.fill_between(lam, a_new, 1e3, color="C0", alpha=0.10, zorder=0)

    ax.plot([lam[i_old]], [a_old[i_old]], "o", color="0.2", ms=6, zorder=5)
    ax.plot([lam[i_new]], [a_new[i_new]], "o", color="C0", ms=6, zorder=5)

    ax.set_xlim(1e5, 1e12)
    ax.set_ylim(1e-11, 1e-3)
    ax.set_xlabel(r"$\lambda$ [m]")
    ax.set_ylabel(r"$|\alpha|$")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    def lam_to_m(l):
        l = np.asarray(l, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return C.HBAR_EV_S * C.C_M_S / l

    secax = ax.secondary_xaxis("top", functions=(lam_to_m, lam_to_m))
    secax.set_xlabel(r"$m_*$ [eV]")

    ax.legend(loc="lower right", frameon=True)
    ax.set_title("Juno fifth-force constraint: Durante 2020 vs Kaspi 2023")
    fig.tight_layout()
    out = "figure4_comparison.png"
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
