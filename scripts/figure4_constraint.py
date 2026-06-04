"""
JHEP01(2025)098 Figure 4: 95% C.L. upper bound on |alpha| vs fifth-force range lambda.

Combines the data-side uncertainty sigma_omega (eq. 3.5) with the model signal
|⟨Δω⟩| at alpha = 1 (eqs. 2.3–2.5):

    |⟨Δω⟩| ≤ 2 sigma_omega  =>  alpha_max(lambda) = 2 sigma_omega / |⟨Δω⟩|_{alpha=1}

Usage (from repository root):

    python scripts/figure4_constraint.py
    python scripts/figure4_constraint.py --show
    python scripts/figure4_constraint.py --quick --n-lambda 30
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import constants as const  # noqa: E402
from src.fifth_force_precession import alpha_upper_bound, constraint_curve  # noqa: E402
from src.gravity_precession import default_orbit, sigma_omega  # noqa: E402

LAMBDA_MIN_M = 1e6
LAMBDA_MAX_M = 1e12


def _integration_kwargs(*, quick=False):
    if quick:
        return dict(n_r=40, n_theta=24, n_steps=512)
    return dict(n_r=60, n_theta=40, n_steps=2048)


def plot_figure4(
    *,
    orbit=None,
    quick=False,
    n_lambda=40,
    lambda_min=LAMBDA_MIN_M,
    lambda_max=LAMBDA_MAX_M,
):
    orbit = orbit or default_orbit()
    integ = _integration_kwargs(quick=quick)
    sw = sigma_omega(orbit)
    print(f"sigma_omega = {sw:.6e} rad/orbit  (95% limit = {2 * sw:.6e})")

    lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_lambda)
    print(f"Scanning {n_lambda} lambda values ({lambda_min:.0e} to {lambda_max:.0e} m) ...")

    lam_arr, alpha_arr = constraint_curve(
        lambdas, sw, orbit=orbit, confidence_factor=const.CONFIDENCE_SIGMA_FACTOR, **integ
    )

    for i, (lam, alpha) in enumerate(zip(lam_arr, alpha_arr)):
        if (i + 1) % max(1, n_lambda // 5) == 0 or i == 0:
            print(f"  {i + 1}/{n_lambda}: lambda = {lam:.3e} m, alpha_max = {alpha:.3e}")

    finite = np.isfinite(alpha_arr) & (alpha_arr > 0)
    if np.any(finite):
        idx_min = int(np.argmin(alpha_arr[finite]))
        lam_at_min = float(lam_arr[finite][idx_min])
        alpha_at_min = float(alpha_arr[finite][idx_min])
        print(
            f"Tightest bound: alpha_max = {alpha_at_min:.3e} at "
            f"lambda = {lam_at_min:.3e} m ({lam_at_min / const.R_JUPITER:.2f} R_X)"
        )

    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)

    plot_mask = finite
    ax.loglog(
        lam_arr[plot_mask],
        alpha_arr[plot_mask],
        "k-",
        lw=2,
        label="Juno (this work)",
    )
    ax.axvline(
        const.R_JUPITER,
        color="0.45",
        ls="--",
        lw=1.2,
        label=r"$\lambda = R_X$",
    )

    ax.set_xlabel(r"fifth-force range $\lambda$ [m]")
    ax.set_ylabel(r"95% C.L. upper bound on $|\alpha|$")
    ax.set_title("Figure 4: fifth-force constraint (Section 4)")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel(r"mediator mass $m_*$ [eV]")
    tick_lams = np.array([1e6, 1e8, 1e10])
    ax2.set_xticks(tick_lams)
    ax2.set_xticklabels(
        [f"{const.MEDIATOR_MASS_PREFACTOR_EV / lam:.2e}" for lam in tick_lams]
    )

    fig.suptitle(
        rf"$\sigma_\omega = {sw:.2e}$ rad orbit$^{{-1}}$; "
        rf"$|\langle\Delta\omega\rangle| \leq {const.CONFIDENCE_SIGMA_FACTOR:.0f}\sigma_\omega$",
        fontsize=10,
        y=1.02,
    )

    return fig, ax, lam_arr, alpha_arr, sw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        type=Path,
        default=ROOT / "output" / "figure4_constraint.png",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--quick", action="store_true", help="coarser volume/orbit grids")
    parser.add_argument("--n-lambda", type=int, default=35)
    parser.add_argument("--lambda-min", type=float, default=LAMBDA_MIN_M)
    parser.add_argument("--lambda-max", type=float, default=LAMBDA_MAX_M)
    args = parser.parse_args()

    fig, _, _, _, _ = plot_figure4(
        quick=args.quick,
        n_lambda=args.n_lambda,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
    )

    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Saved {args.save}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
