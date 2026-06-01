"""
Gravity-field covariance for JHEP01(2025)098 eq. (3.5).

    σ²_ω = (∂⟨Δω_g⟩/∂J)ᵀ C_J (∂⟨Δω_g⟩/∂J)

C_J is the 10×10 **covariance** of J = [μ, J₂, …, J₁₀] from Durante et al. 2020
Supporting Information Data Set S2 (``context/durante_supporting/covariancematrix.txt``).
Diagonal entries are **variances** C_ii = σ_i² in the parameters' physical units
(e.g. GM variance in (km³ s⁻²)², J_l variance dimensionless²)—not 1 and not
dimensionless correlation coefficients. A separate ``correlation_matrix()`` exists
only for optional inspection; eq. (3.5) and :data:`COVARIANCE_JHEP_ONE_SIGMA` use
covariance throughout.

S2 is quoted for fully normalized Stokes coefficients; this module extracts the
JHEP zonal block and maps to unnormalized J_l (Table 2 convention) via
J_l^unnorm = √(2l + 1) J_l^norm.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from .constants import N_HARMONICS

# Durante GRL 47, e2019GL086572 — Supporting Information Data Set S2.
COVARIANCE_SOURCE_DOI: str = "10.1029/2019GL086572"

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
COVARIANCE_MATRIX_PATH: Path = (
    REPO_ROOT / "context" / "durante_supporting" / "covariancematrix.txt"
)

# Row/column order for the JHEP parameter vector (eq. 3.5, N = 10).
PARAMETER_LABELS: tuple[str, ...] = (
    "GM",
    "J[2]",
    "J[3]",
    "J[4]",
    "J[5]",
    "J[6]",
    "J[7]",
    "J[8]",
    "J[9]",
    "J[10]",
)

ZONAL_DEGREES: tuple[int, ...] = tuple(range(2, N_HARMONICS + 1))

# Physical units for each entry of J (variances are units squared).
PARAMETER_UNITS: tuple[str, ...] = (
    "km^3 s^-2",
    "1",
    "1",
    "1",
    "1",
    "1",
    "1",
    "1",
    "1",
    "1",
)

# Durante Table 2 reports 3σ formal uncertainties; S2 diagonal matches σ²_{3σ}.
DURANTE_SIGMA_LEVEL: int = 3


def zonal_unnorm_scale(degree: int) -> float:
    """√(2l + 1): Jacobian from normalized to unnormalized zonal J_l."""
    return (2 * degree + 1) ** 0.5


def _parameter_scales() -> tuple[float, ...]:
    """Diagonal scales for [μ, J₂, …, J_N]; μ is unchanged."""
    return (1.0,) + tuple(zonal_unnorm_scale(degree) for degree in ZONAL_DEGREES)


def _parse_covariance_file(path: Path) -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...]]:
    lines = path.read_text().strip().splitlines()
    names = tuple(line.split()[0] for line in lines)
    matrix = tuple(
        tuple(float(value) for value in line.split()[1:])
        for line in lines
    )
    return names, matrix


def load_covariance_normalized(
    path: Path | None = None,
) -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...]]:
    """Full S2 covariance (normalized Stokes coefficients) with parameter names."""
    return _parse_covariance_file(path or COVARIANCE_MATRIX_PATH)


def load_covariance_jhep(
    path: Path | None = None,
    *,
    max_degree: int = N_HARMONICS,
) -> tuple[tuple[float, ...], ...]:
    """
    JHEP block C_J for J = [μ, J₂, …, J_max_degree], unnormalized J_l.

    Entries are at the Durante Table 2 reporting level (3σ formal on the diagonal).
    Use :data:`COVARIANCE_JHEP_ONE_SIGMA` in eq. (3.5).
    """
    if max_degree < 2:
        raise ValueError("max_degree must be at least 2")
    labels = ("GM",) + tuple(f"J[{degree}]" for degree in range(2, max_degree + 1))
    names, matrix = load_covariance_normalized(path)
    name_to_index = {name: index for index, name in enumerate(names)}
    indices = tuple(name_to_index[label] for label in labels)
    scales = _parameter_scales()[: len(labels)]

    block: list[tuple[float, ...]] = []
    for i, row in enumerate(indices):
        block.append(
            tuple(
                matrix[row][col] * scales[i] * scales[j]
                for j, col in enumerate(indices)
            )
        )
    return tuple(block)


def scale_covariance_to_one_sigma(
    covariance: Sequence[Sequence[float]],
    *,
    from_sigma_level: int = DURANTE_SIGMA_LEVEL,
) -> tuple[tuple[float, ...], ...]:
    """Convert a covariance quoted at kσ to 1σ (divide all entries by k²)."""
    factor = float(from_sigma_level * from_sigma_level)
    return tuple(tuple(value / factor for value in row) for row in covariance)


def diagonal_sigmas(
    covariance: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """Parameter 1σ uncertainties √(C_ii) in :data:`PARAMETER_LABELS` order."""
    return tuple(math.sqrt(row[index]) for index, row in enumerate(covariance))


def quadratic_form(
    gradient: Sequence[float],
    covariance: Sequence[Sequence[float]],
) -> float:
    """Evaluate gᵀ C g for eq. (3.5)."""
    size = len(gradient)
    total = 0.0
    for i in range(size):
        row = covariance[i]
        gi = gradient[i]
        for j in range(size):
            total += gi * row[j] * gradient[j]
    return total


def variance_delta_omega(
    gradient: Sequence[float],
    covariance: Sequence[Sequence[float]] | None = None,
) -> float:
    """
    Variance σ²_ω of ⟨Δω_g⟩ from eq. (3.5) [rad² orbit⁻²].

    ``gradient`` is [∂⟨Δω_g⟩/∂μ, ∂⟨Δω_g⟩/∂J₂, …] in the same units as the
    gravity-field parameters in C_J.
    """
    cov = covariance if covariance is not None else COVARIANCE_JHEP_ONE_SIGMA
    size = len(gradient)
    block = tuple(tuple(row[col] for col in range(size)) for row in cov[:size])
    return quadratic_form(gradient, block)


def sigma_delta_omega(
    gradient: Sequence[float],
    covariance: Sequence[Sequence[float]] | None = None,
) -> float:
    """Standard deviation σ_ω = √(σ²_ω) [rad orbit⁻¹]."""
    return math.sqrt(variance_delta_omega(gradient, covariance))


# 10×10 at Durante 3σ level (diagonal √C_ii matches Table 2 formal errors).
COVARIANCE_JHEP: tuple[tuple[float, ...], ...] = load_covariance_jhep()

# 1σ matrix for eq. (3.5).
COVARIANCE_JHEP_ONE_SIGMA: tuple[tuple[float, ...], ...] = scale_covariance_to_one_sigma(
    COVARIANCE_JHEP
)


def as_numpy(
    covariance: Sequence[Sequence[float]] | None = None,
):
    """Return C_J as a NumPy array (optional dependency)."""
    import numpy as np

    block = covariance if covariance is not None else COVARIANCE_JHEP_ONE_SIGMA
    return np.asarray(block, dtype=float)


def load_covariance_jhep_normalized_block(
    path: Path | None = None,
    *,
    max_degree: int = N_HARMONICS,
) -> tuple[tuple[float, ...], ...]:
    """JHEP 10x10 block from S2 *before* unnormalized scaling (for debugging)."""
    if max_degree < 2:
        raise ValueError("max_degree must be at least 2")
    labels = ("GM",) + tuple(f"J[{degree}]" for degree in range(2, max_degree + 1))
    names, matrix = load_covariance_normalized(path)
    name_to_index = {name: index for index, name in enumerate(names)}
    indices = tuple(name_to_index[label] for label in labels)
    return tuple(
        tuple(matrix[row][col] for col in indices) for row in indices
    )


def correlation_matrix(
    covariance: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    """
    Pearson correlation ρ_ij = C_ij / √(C_ii C_jj).

    For debugging shape/sign of off-diagonals only. Do **not** use in eq. (3.5);
    diagonal is always 1 here by construction, unlike :data:`COVARIANCE_JHEP`.
    """
    size = len(covariance)
    diag = tuple(math.sqrt(covariance[i][i]) for i in range(size))
    block: list[tuple[float, ...]] = []
    for i in range(size):
        row_values: list[float] = []
        for j in range(size):
            denom = diag[i] * diag[j]
            row_values.append(covariance[i][j] / denom if denom else float("nan"))
        block.append(tuple(row_values))
    return tuple(block)


def _format_value(value: float, *, precision: int = 4) -> str:
    if value == 0.0:
        return "0"
    abs_val = abs(value)
    if abs_val >= 1e4 or (abs_val > 0 and abs_val < 1e-3):
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"


def format_matrix(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    title: str = "",
    precision: int = 4,
    max_label_width: int = 8,
) -> str:
    """Return a fixed-width, labeled table for terminal inspection."""
    label_width = min(
        max(len(label) for label in labels),
        max_label_width,
    )
    col_headers = [label[:label_width].rjust(label_width) for label in labels]
    cell_width = max(10, precision + 6)
    header = " " * (label_width + 2) + " ".join(h.rjust(cell_width) for h in col_headers)

    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("-" * len(title))
    lines.append(header)
    for label, row in zip(labels, matrix):
        row_label = label[:label_width].ljust(label_width)
        cells = " ".join(
            _format_value(value, precision=precision).rjust(cell_width) for value in row
        )
        lines.append(f"{row_label}  {cells}")
    return "\n".join(lines)


def print_matrix(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[str] | None = None,
    *,
    title: str = "",
    precision: int = 4,
) -> None:
    """Print a labeled matrix to stdout."""
    row_labels = labels if labels is not None else tuple(str(i) for i in range(len(matrix)))
    print(format_matrix(matrix, row_labels, title=title, precision=precision))
    print()


def verify_covariance_not_correlation(
    covariance: Sequence[Sequence[float]],
    labels: Sequence[str] = PARAMETER_LABELS,
) -> None:
    """
    Raise if a matrix looks like a correlation matrix (unit diagonal on J_l block).

    GM can have C_ii ~ 70 at 3σ; J_l diagonals must be tiny positive variances.
    """
    size = len(covariance)
    for index in range(size):
        variance = covariance[index][index]
        if variance <= 0.0:
            raise ValueError(f"{labels[index]}: non-positive diagonal {variance}")
        if labels[index] == "GM":
            continue
        if abs(variance - 1.0) < 1e-6:
            raise ValueError(
                f"{labels[index]}: diagonal ~ 1 looks like correlation, not covariance"
            )


def print_units_legend() -> None:
    """Explain units of C_ij (covariance, not correlation)."""
    print("Covariance units (C_ij has units of parameter_i x parameter_j):")
    print("-" * 56)
    for label, unit in zip(PARAMETER_LABELS, PARAMETER_UNITS):
        var_unit = f"({unit})^2" if unit != "1" else "dimensionless^2"
        print(f"  {label:8s}  parameter [{unit}]   ->  C_ii [{var_unit}]")
    print()


def print_raw_file_diagonal(
    path: Path | None = None,
    *,
    labels: Sequence[str] = PARAMETER_LABELS,
) -> None:
    """Print C_ii straight from covariancematrix.txt (normalized S2, JHEP subset)."""
    block = load_covariance_jhep_normalized_block(path)
    print("Diagonal variances C_ii from covariancematrix.txt (JHEP labels, normalized):")
    print("-" * 56)
    for index, label in enumerate(labels):
        print(f"  {label:8s}  C_ii = {block[index][index]:.6e}")
    print("  (Compare to row self-entry in the text file; e.g. GM -> 7.046e+01)")
    print()


def print_diagonal_summary(
    covariance: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    title: str,
    table2_sigma_3: Sequence[float] | None = None,
) -> None:
    """Print sqrt(C_ii) and optional Table 2 comparison."""
    print(title)
    print("-" * len(title))
    sigmas = diagonal_sigmas(covariance)
    for index, label in enumerate(labels):
        line = f"  {label:8s}  sqrt(C_ii) = {sigmas[index]:.6e}"
        if table2_sigma_3 is not None and index < len(table2_sigma_3):
            expected = table2_sigma_3[index]
            rel = abs(sigmas[index] - expected) / expected if expected else 0.0
            status = "ok" if rel < 0.02 else "CHECK"
            line += f"  (Table2 3-sigma: {expected:.6e}, rel err {rel:.2%}) [{status}]"
        print(line)
    print()


def debug_display(
    *,
    show_normalized: bool = True,
    show_jhep_3sigma: bool = True,
    show_jhep_1sigma: bool = True,
    show_correlation: bool = False,
    show_scales: bool = True,
    show_validation: bool = True,
    show_raw_diagonal: bool = True,
    precision: int = 4,
    path: Path | None = None,
) -> None:
    """Print **covariance** matrices for manual inspection (not correlation by default)."""
    labels = PARAMETER_LABELS
    names, full_normalized = load_covariance_normalized(path)
    norm_block = load_covariance_jhep_normalized_block(path)
    jhep_3 = load_covariance_jhep(path)
    jhep_1 = scale_covariance_to_one_sigma(jhep_3)
    verify_covariance_not_correlation(jhep_3, labels)
    verify_covariance_not_correlation(jhep_1, labels)

    print("=" * 72)
    print("gravity_covariance debug display  [COVARIANCE C, not correlation]")
    print("=" * 72)
    print(f"Source file : {path or COVARIANCE_MATRIX_PATH}")
    print(f"DOI         : {COVARIANCE_SOURCE_DOI}")
    n_s2 = len(full_normalized)
    print(f"S2 size     : {n_s2} x {len(full_normalized[0]) if n_s2 else 0} parameters")
    print(f"JHEP block  : {len(labels)} x {len(labels)} ({', '.join(labels)})")
    print(
        "Note        : Diagonal C_ii are variances (e.g. GM ~ 70 at 3sigma), not 1."
    )
    print(
        "              Use --show-correlation only if you want rho_ij with diag=1."
    )
    print()
    print_units_legend()

    if show_raw_diagonal:
        print_raw_file_diagonal(path, labels=labels)

    if show_scales:
        scales = _parameter_scales()
        print("Unnormalized scaling factors (mu=1, J_l *= sqrt(2l+1)):")
        print("-" * 56)
        for label, scale in zip(labels, scales):
            print(f"  {label:8s}  {scale:.6f}")
        print()

    if show_normalized:
        print_matrix(
            norm_block,
            labels,
            title="COVARIANCE: JHEP block from S2 (normalized Stokes, before sqrt(2l+1) scaling)",
            precision=precision,
        )

    if show_jhep_3sigma:
        print_matrix(
            jhep_3,
            labels,
            title="COVARIANCE C_J at Durante 3-sigma (unnormalized J_l; diag = sigma_3^2)",
            precision=precision,
        )

    if show_jhep_1sigma:
        print_matrix(
            jhep_1,
            labels,
            title="COVARIANCE C_J at 1-sigma (C/9; used in JHEP eq. 3.5)",
            precision=precision,
        )

    if show_correlation:
        print(
            ">>> Optional correlation view (rho_ii = 1 by definition; NOT used in eq. 3.5)"
        )
        print()
        print_matrix(
            correlation_matrix(jhep_3),
            labels,
            title="CORRELATION rho from 3-sigma COVARIANCE (diagonal = 1)",
            precision=precision,
        )
        print_matrix(
            correlation_matrix(jhep_1),
            labels,
            title="CORRELATION rho from 1-sigma COVARIANCE (diagonal = 1)",
            precision=precision,
        )

    if show_validation:
        table2_sigma_3 = (
            8.4,
            0.0017e-6,
            0.0033e-6,
            0.0024e-6,
            0.0042e-6,
            0.0067e-6,
            0.012e-6,
            0.021e-6,
            0.036e-6,
            0.065e-6,
        )
        print_diagonal_summary(
            jhep_3,
            labels,
            title="Diagonal check vs Durante Table 2 (3-sigma formal)",
            table2_sigma_3=table2_sigma_3,
        )
        print_diagonal_summary(
            jhep_1,
            labels,
            title="1-sigma diagonal (sqrt of COVARIANCE_JHEP_ONE_SIGMA)",
        )

    print("Off-diagonal spot check (3-sigma C_J, upper triangle):")
    print("-" * 56)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            value = jhep_3[i][j]
            if value != 0.0:
                print(f"  C[{labels[i]}, {labels[j]}] = {value:.6e}")
    print()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and display Jupiter gravity covariance matrices (JHEP eq. 3.5)."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Override path to covariancematrix.txt",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Significant digits in matrix cells (default: 4)",
    )
    parser.add_argument(
        "--only",
        choices=("normalized", "3sigma", "1sigma", "correlation", "scales", "validation"),
        action="append",
        dest="only",
        help="Show only selected sections (repeatable; default: all)",
    )
    parser.add_argument(
        "--show-correlation",
        action="store_true",
        help="Also print correlation matrices (diag=1; not used in eq. 3.5)",
    )
    parser.add_argument(
        "--save-numpy",
        type=Path,
        metavar="FILE",
        help="Also write 1-sigma C_J to a .npy file",
    )
    args = parser.parse_args()

    if args.only:
        mapping = {
            "normalized": "show_normalized",
            "3sigma": "show_jhep_3sigma",
            "1sigma": "show_jhep_1sigma",
            "correlation": "show_correlation",
            "scales": "show_scales",
            "validation": "show_validation",
        }
        show = {flag: False for flag in mapping.values()}
        for key in args.only:
            show[mapping[key]] = True
    else:
        show = {
            "show_normalized": True,
            "show_jhep_3sigma": True,
            "show_jhep_1sigma": True,
            "show_correlation": args.show_correlation,
            "show_scales": True,
            "show_validation": True,
            "show_raw_diagonal": True,
        }

    debug_display(
        path=args.path,
        precision=args.precision,
        **show,
    )

    if args.save_numpy:
        import numpy as np

        out = as_numpy(COVARIANCE_JHEP_ONE_SIGMA)
        args.save_numpy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_numpy, out)
        print(f"Wrote 1-sigma C_J to {args.save_numpy}")


if __name__ == "__main__":
    main()
