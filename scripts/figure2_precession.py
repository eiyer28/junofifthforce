"""
JHEP01(2025)098 Figure 2: orbit-averaged fifth-force precession ⟨Δω⟩.

Left: ⟨Δω⟩ vs α at fixed λ = 10⁻¹ A.U. and 10⁻² A.U. (eqs. 2.3–2.5).
Right: ⟨Δω⟩ vs λ at fixed α = 10⁻⁷ and 10⁻⁶; dashed line at λ = R_X.

Usage (from repository root):

    python scripts/figure2_precession.py
    python scripts/figure2_precession.py --show
    python scripts/figure2_precession.py --quick
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
from src.fifth_force_precession import mean_fifth_force_precession_unit  # noqa: E402
from src.gravity_precession import default_orbit  # noqa: E402

# Paper Figure 2 defaults (Section 2)
LAMBDA_FIXED_AU = (0.1, 0.01)
ALPHA_FIXED = (1e-7, 1e-6)
ALPHA_MIN = 1e-9
ALPHA_MAX = 1e-7
LAMBDA_MIN_M = 1e6
LAMBDA_MAX_M = 1e12


def _integration_kwargs(*, quick=False):
    if quick:
        return dict(n_r=40, n_theta=24, n_steps=512)
    return dict(n_r=60, n_theta=40, n_steps=2048)


def _abs_delta_omega_unit(lam_m, orbit, integ_kwargs):
    """|⟨Δω⟩| at α = 1 [rad orbit⁻¹]."""
    return abs(mean_fifth_force_precession_unit(lam_m, orbit=orbit, **integ_kwargs))


def plot_figure2(
    *,
    orbit=None,
    quick=False,
    n_alpha=30,
    n_lambda=40,
    alpha_min=ALPHA_MIN,
    alpha_max=ALPHA_MAX,
    lambda_min=LAMBDA_MIN_M,
    lambda_max=LAMBDA_MAX_M,
):
    orbit = orbit or default_orbit()
    integ = _integration_kwargs(quick=quick)
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha)
    lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_lambda)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    # --- Left: ⟨Δω⟩ vs α at fixed λ ---
    for lam_au in LAMBDA_FIXED_AU:
        lam_m = lam_au * const.AU
        dw_unit = _abs_delta_omega_unit(lam_m, orbit, integ)
        dw = dw_unit * alphas
        ax_left.loglog(
            alphas,
            dw,
            lw=2,
            label=rf"$\lambda = 10^{{{int(round(np.log10(lam_au)))}}}$ A.U.",
        )
        print(f"lambda = {lam_au:.0e} A.U. ({lam_m:.3e} m): |<dw>/alpha| = {dw_unit:.3e} rad/orbit")

    ax_left.set_xlabel(r"$\alpha$")
    ax_left.set_ylabel(r"$|\langle\Delta\omega\rangle|$ [rad orbit$^{-1}$]")
    ax_left.set_title(r"$\langle\Delta\omega\rangle$ vs $\alpha$")
    ax_left.legend(loc="lower right")
    ax_left.grid(True, which="both", alpha=0.3)

    # --- Right: ⟨Δω⟩ vs λ at fixed α ---
    print(f"Scanning {n_lambda} lambda values ({lambda_min:.0e} to {lambda_max:.0e} m) ...")
    for alpha in ALPHA_FIXED:
        dw_curve = np.empty(n_lambda)
        for i, lam_m in enumerate(lambdas):
            dw_unit = _abs_delta_omega_unit(float(lam_m), orbit, integ)
            dw_curve[i] = dw_unit * alpha
            if (i + 1) % max(1, n_lambda // 5) == 0:
                print(f"  alpha = {alpha:.0e}: {i + 1}/{n_lambda} lambda points")

        ax_right.loglog(
            lambdas,
            dw_curve,
            lw=2,
            label=rf"$\alpha = 10^{{{int(round(np.log10(alpha)))}}}$",
        )

    ax_right.axvline(
        const.R_JUPITER,
        color="0.45",
        ls="--",
        lw=1.5,
        label=r"$\lambda = R_X$",
    )
    ax_right.set_xlabel(r"fifth-force range $\lambda$ [m]")
    ax_right.set_ylabel(r"$|\langle\Delta\omega\rangle|$ [rad orbit$^{-1}$]")
    ax_right.set_title(r"$\langle\Delta\omega\rangle$ vs $\lambda$")
    ax_right.legend(loc="upper right")
    ax_right.grid(True, which="both", alpha=0.3)

    ax2 = ax_right.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax_right.get_xlim())
    ax2.set_xlabel(r"mediator mass $m_*$ [eV]")
    tick_lams = np.array([1e6, 1e8, 1e10])
    ax2.set_xticks(tick_lams)
    ax2.set_xticklabels(
        [f"{const.MEDIATOR_MASS_PREFACTOR_EV / lam:.2e}" for lam in tick_lams]
    )

    fig.suptitle("Figure 2: fifth-force precession (eqs. 2.3–2.5)", fontsize=12, y=1.02)

    # Peak location for α = 10⁻⁷ (paper: λ ≈ 0.4 R_X)
    dw_units = np.array([_abs_delta_omega_unit(float(lam), orbit, integ) for lam in lambdas])
    peak_idx = int(np.argmax(dw_units))
    peak_lam = float(lambdas[peak_idx])
    print(
        f"Peak |<dw>/alpha| at lambda = {peak_lam:.3e} m "
        f"({peak_lam / const.R_JUPITER:.2f} R_X, "
        f"{peak_lam / const.AU:.2e} A.U.)"
    )

    return fig, (ax_left, ax_right), alphas, lambdas, dw_units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        type=Path,
        default=ROOT / "output" / "figure2_precession.png",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--quick", action="store_true", help="coarser volume/orbit grids")
    parser.add_argument("--n-alpha", type=int, default=30)
    parser.add_argument("--n-lambda", type=int, default=35)
    parser.add_argument("--alpha-min", type=float, default=ALPHA_MIN)
    parser.add_argument("--alpha-max", type=float, default=ALPHA_MAX)
    parser.add_argument("--lambda-min", type=float, default=LAMBDA_MIN_M)
    parser.add_argument("--lambda-max", type=float, default=LAMBDA_MAX_M)
    args = parser.parse_args()

    fig, _, _, _, _ = plot_figure2(
        quick=args.quick,
        n_alpha=args.n_alpha,
        n_lambda=args.n_lambda,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
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
