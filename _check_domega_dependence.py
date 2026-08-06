"""Confirm why only low-degree J's set sigma_omega: |dΔω/dp| vs σ_p competition.

Hypothesis (Lingfeng): Δω depends more strongly on low-ranked gravity parameters
(GM, J2, ...), which are also better measured. Higher-N uncertainties may grow, but
not fast enough to offset the geometric decay of |∂Δω/∂J_l| ~ (R_J/a)^l, so the
product |g_l| σ_l collapses and sigma_omega plateaus.

Run:  python _check_domega_dependence.py
"""

import numpy as np

from constants import A_KM, ECC, INC, OMEGA, R_J_KM
from covariance import covariance_slice, covariance_slice_new
from precession_gravity import gradient_vector
from precession_zonal_gauss import domega_dJl as domega_dJl_gauss


def report_budget(name, cov, apply_normalization, n_max=10, use_gauss=False):
    """Print |g|, σ_p, and diagonal product |g|σ for [GM, J2..Jn]."""
    if use_gauss:
        # J-only, raw basis (matches cov_220824-style blocks)
        g = np.array([
            domega_dJl_gauss(l, A_KM, ECC, INC, OMEGA, R_J_KM)
            for l in range(2, n_max + 1)
        ])
        labels = [f"J{l}" for l in range(2, n_max + 1)]
        # assume cov is already J-only
        C = cov[: n_max - 1, : n_max - 1]
    else:
        g = gradient_vector(
            A_KM, ECC, INC, OMEGA, R_J_KM, n_max,
            apply_normalization=apply_normalization,
        )
        labels = ["GM"] + [f"J{l}" for l in range(2, n_max + 1)]
        C = cov[:n_max, :n_max]

    sig = np.sqrt(np.diag(C))
    term = np.abs(g) * sig
    var_full = float(g @ C @ g)
    var_diag = float(np.sum(g**2 * np.diag(C)))

    print(f"\n{'=' * 78}")
    print(f"{name}  (apply_normalization={apply_normalization}, n_max={n_max})")
    print(f"{'=' * 78}")
    print(f"{'param':>6} {'|g|=|dΔω/dp|':>14} {'σ_p':>12} {'|g|σ':>12} "
          f"{'|g|/|g_J2|':>11} {'σ/σ_J2':>9} {'frac|g|σ':>10}")
    g_j2 = abs(g[labels.index("J2")]) if "J2" in labels else abs(g[0])
    s_j2 = sig[labels.index("J2")] if "J2" in labels else sig[0]
    for lab, gi, si, ti in zip(labels, g, sig, term):
        print(f"{lab:>6} {abs(gi):>14.4e} {si:>12.4e} {ti:>12.4e} "
              f"{abs(gi)/g_j2:>11.4f} {si/s_j2:>9.3f} {ti/term.sum():>9.1%}")

    print(f"\n  sigma_omega (full, with correlations) = {np.sqrt(max(var_full, 0)):.4e}")
    print(f"  sigma_omega (diag-only)               = {np.sqrt(max(var_diag, 0)):.4e}")
    print(f"  cross/diag                            = {(var_full - var_diag)/var_diag:+.3f}")
    print(f"  R_J/a                                 = {R_J_KM/A_KM:.4f}  "
          f"(geometric factor per degree)")

    # cumulative truncation curve
    print(f"\n  cumulative sigma_omega as N grows:")
    if use_gauss:
        for n in range(2, n_max + 1):
            m = n - 1
            s = float(np.sqrt(g[:m] @ C[:m, :m] @ g[:m]))
            print(f"    N={n:2d}: {s:.4e}")
    else:
        for k in range(1, n_max + 1):  # 1=GM only, 2=GM+J2, ...
            s = float(np.sqrt(g[:k] @ C[:k, :k] @ g[:k]))
            lab = labels[k - 1]
            print(f"    through {lab:>3}: {s:.4e}")


def main():
    print("Lingfeng check: does |dΔω/dJ_l| decay faster than σ(J_l) grows?")
    print(f"orbit: a={A_KM:.1f} km, e={ECC:.4f}, i={INC}, ω={OMEGA}")

    # Paper reproduction convention for Durante/Kaspi (both fully-normalised cov)
    report_budget(
        "Durante 2020 [GM,J2..J10]",
        covariance_slice(10),
        apply_normalization=False,
        n_max=10,
    )
    report_budget(
        "Kaspi 2023 [GM,J2..J10]",
        covariance_slice_new(10),
        apply_normalization=False,
        n_max=10,
    )

    # Physically consistent: normalised gradient x normalised cov
    report_budget(
        "Durante 2020 (basis-consistent)",
        covariance_slice(10),
        apply_normalization=True,
        n_max=10,
    )


if __name__ == "__main__":
    main()
