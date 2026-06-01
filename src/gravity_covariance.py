"""
Gravity-field covariance for JHEP01(2025)098 eq. (3.5).

    σ²_ω = (∂⟨Δω_g⟩/∂J)ᵀ C_J (∂⟨Δω_g⟩/∂J)

C_J is the 10×10 covariance of J = [μ, J₂, …, J₁₀] from Durante et al. 2020
Supporting Information Data Set S2 (``context/durante_supporting/covariancematrix.txt``).
Diagonal entries are variances C_ii = σ_i² in the parameters' physical units—not 1.
Eq. (3.5) uses :data:`COVARIANCE_JHEP_ONE_SIGMA` (1σ covariance, not correlation).

S2 is for fully normalized Stokes coefficients; the JHEP block maps to unnormalized
J_l via J_l^unnorm = √(2l + 1) J_l^norm.
"""

import math
from pathlib import Path

from .constants import N_HARMONICS

COVARIANCE_SOURCE_DOI = "10.1029/2019GL086572"

REPO_ROOT = Path(__file__).resolve().parents[1]
COVARIANCE_MATRIX_PATH = REPO_ROOT / "context" / "durante_supporting" / "covariancematrix.txt"

PARAMETER_LABELS = (
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

ZONAL_DEGREES = tuple(range(2, N_HARMONICS + 1))
DURANTE_SIGMA_LEVEL = 3


def zonal_unnorm_scale(degree):
    """√(2l + 1): Jacobian from normalized to unnormalized zonal J_l."""
    return (2 * degree + 1) ** 0.5


def _parameter_scales():
    return (1.0,) + tuple(zonal_unnorm_scale(degree) for degree in ZONAL_DEGREES)


def _parse_covariance_file(path):
    lines = path.read_text().strip().splitlines()
    names = tuple(line.split()[0] for line in lines)
    matrix = tuple(
        tuple(float(value) for value in line.split()[1:])
        for line in lines
    )
    return names, matrix


def load_covariance_normalized(path=None):
    """Full S2 covariance (normalized Stokes coefficients) with parameter names."""
    return _parse_covariance_file(path or COVARIANCE_MATRIX_PATH)


def load_covariance_jhep(path=None, *, max_degree=N_HARMONICS):
    """
    JHEP block C_J for J = [μ, J₂, …, J_max_degree], unnormalized J_l.

    Entries are at Durante Table 2 reporting level (3σ formal on the diagonal).
    Use COVARIANCE_JHEP_ONE_SIGMA in eq. (3.5).
    """
    if max_degree < 2:
        raise ValueError("max_degree must be at least 2")
    labels = ("GM",) + tuple(f"J[{degree}]" for degree in range(2, max_degree + 1))
    names, matrix = load_covariance_normalized(path)
    name_to_index = {name: index for index, name in enumerate(names)}
    indices = tuple(name_to_index[label] for label in labels)
    scales = _parameter_scales()[: len(labels)]

    block = []
    for i, row in enumerate(indices):
        block.append(
            tuple(
                matrix[row][col] * scales[i] * scales[j]
                for j, col in enumerate(indices)
            )
        )
    return tuple(block)


def scale_covariance_to_one_sigma(covariance, *, from_sigma_level=DURANTE_SIGMA_LEVEL):
    """Convert a covariance quoted at kσ to 1σ (divide all entries by k²)."""
    factor = float(from_sigma_level * from_sigma_level)
    return tuple(tuple(value / factor for value in row) for row in covariance)


def quadratic_form(gradient, covariance):
    """Evaluate gᵀ C g for eq. (3.5)."""
    size = len(gradient)
    total = 0.0
    for i in range(size):
        row = covariance[i]
        gi = gradient[i]
        for j in range(size):
            total += gi * row[j] * gradient[j]
    return total


def variance_delta_omega(gradient, covariance=None):
    """
    Variance σ²_ω of ⟨Δω_g⟩ from eq. (3.5) [rad² orbit⁻²].

    gradient is [∂⟨Δω_g⟩/∂μ, ∂⟨Δω_g⟩/∂J₂, …] in the same units as C_J.
    """
    cov = covariance if covariance is not None else COVARIANCE_JHEP_ONE_SIGMA
    size = len(gradient)
    block = tuple(tuple(row[col] for col in range(size)) for row in cov[:size])
    return quadratic_form(gradient, block)


def sigma_delta_omega(gradient, covariance=None):
    """Standard deviation σ_ω = √(σ²_ω) [rad orbit⁻¹]."""
    return math.sqrt(variance_delta_omega(gradient, covariance))


COVARIANCE_JHEP = load_covariance_jhep()
COVARIANCE_JHEP_ONE_SIGMA = scale_covariance_to_one_sigma(COVARIANCE_JHEP)
