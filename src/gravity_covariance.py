"""
Gravity-field covariance for JHEP01(2025)098 eq. (3.5).

    σ²_ω = (∂⟨Δω_g⟩/∂J)ᵀ C_J (∂⟨Δω_g⟩/∂J)

C_J is the 10×10 **covariance** of J = [μ, J₂, …, J₁₀] from Durante et al. 2020
Supporting Information Data Set S2 (``context/durante_supporting/covariancematrix.txt``).
Diagonal entries are **variances** C_ii = σ_i² in the parameters' physical units
(e.g. GM variance in (km³ s⁻²)², J_l variance dimensionless²)—not 1 and not
dimensionless correlation coefficients. Eq. (3.5) and :data:`COVARIANCE_JHEP_ONE_SIGMA` use
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
