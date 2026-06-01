"""
Physical and analysis constants for reproducing Singh et al., JHEP01(2025)098.

Sources
-------
- Paper-fixed values: JHEP01(2025)098 / arXiv:2409.10616 (Juno fifth-force bounds).
- CODATA 2022: NIST fundamental physical constants (2022 revision).
"""

from __future__ import annotations

# Universally accepted constants fronm CODATA 2022 (NIST)
# https://physics.nist.gov/cuu/Constants/

# Newtonian gravitational constant [m^3 kg^-1 s^-2]
G: float = 6.67430e-11

# Speed of light in vacuum [m s^-1] (exact)
C: float = 299_792_458.0

# Reduced Planck constant [J s]
HBAR: float = 1.054571817e-34

# Planck constant [J s] (exact); included for completeness
H: float = 6.62607015e-34

# Analysis conventions fixed by 2025 Juno fifth force paper (JHEP01(2025)098)

# Jovian equatorial radius [m] (sourced from section 2)
R_JUPITER: float = 7.1e4 * 1e3

# Astronomical unit [m] (sourced from figure 2 caption). We use the same round value as the paper.
AU: float = 1.5e11

# Conservative orbit choice for σ_ω (sourced from section 3): first-orbit elements [rad].
INCLINATION: float = 1.57
ARGUMENT_OF_PERIAPSIS: float = 3.08

# Highest retained zonal harmonic order (sourced from section 3).
N_HARMONICS: int = 10

# 95% confidence level: |⟨Δω⟩| ≤ 2 σ_ω (sourced from section 4).
CONFIDENCE_SIGMA_FACTOR: float = 2.0

# Target σ_ω after error propagation [rad orbit^-1] (Section 3, Figure 3).
# Used to validate the gravity-field uncertainty pipeline, not as an input
# once C_J is available.
SIGMA_OMEGA_TARGET: float = 9.0e-10
SIGMA_OMEGA_MIN: float = 7.1e-10
SIGMA_OMEGA_MAX: float = 9.0e-10

# Figure 3 contour ranges [rad] (Section 3, first–seventeenth perijoves).
INCLINATION_MIN: float = 1.55
INCLINATION_MAX: float = 1.75
ARGUMENT_OF_PERIAPSIS_MIN: float = 2.80
ARGUMENT_OF_PERIAPSIS_MAX: float = 3.10

# Durante Table 2 reference radius [km]; Appendix B R_Jupiter.
R_JUPITER_REF_KM: float = 71492.0

# PJ01 osculating elements for eq. (3.5) / Figure 3 (fixed a, e; vary i, ω).
SEMI_MAJOR_AXIS_KM: float = 1.087e6
ECCENTRICITY: float = 0.824

KM_TO_M: float = 1000.0
TWO_PI: float = 6.283185307179586

# Constants derived from paper-fixed CODATA + geometry for convenience
EV_JOULE: float = 1.602176634e-19
MEDIATOR_MASS_PREFACTOR_EV: float = HBAR * C / EV_JOULE  # m* [eV] = HBAR * C / EV_JOULE * λ [m]
