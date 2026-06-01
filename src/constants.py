"""
Physical and analysis constants for reproducing Singh et al., JHEP01(2025)098.

Sources
-------
- Paper-fixed values: JHEP01(2025)098 / arXiv:2409.10616 (Juno fifth-force bounds).
- CODATA 2022: NIST fundamental physical constants (2022 revision).
"""

# CODATA 2022 (NIST) — https://physics.nist.gov/cuu/Constants/

G = 6.67430e-11  # gravitational constant [m^3 kg^-1 s^-2]
C = 299_792_458.0  # speed of light [m s^-1] (exact)
HBAR = 1.054571817e-34  # reduced Planck constant [J s]
H = 6.62607015e-34  # Planck constant [J s] (exact)

# JHEP01(2025)098 analysis conventions

R_JUPITER = 7.1e4 * 1e3  # Jovian equatorial radius [m]
AU = 1.5e11  # astronomical unit [m]

INCLINATION = 1.57  # conservative σ_ω choice, PJ01 [rad]
ARGUMENT_OF_PERIAPSIS = 3.08  # conservative σ_ω choice, PJ01 [rad]

N_HARMONICS = 10
CONFIDENCE_SIGMA_FACTOR = 2.0  # 95% C.L.: |⟨Δω⟩| ≤ 2 σ_ω

SIGMA_OMEGA_TARGET = 9.0e-10  # JHEP Section 3 validation target [rad orbit^-1]

R_JUPITER_REF_KM = 71492.0  # Durante Table 2 / Appendix B R_X [km]

PDS_PJ1_RANGE_PERI_RJ = 1.058
PDS_PJ1_RANGE_APO_RJ = 113.08
SEMI_MAJOR_AXIS_KM = (
    (PDS_PJ1_RANGE_PERI_RJ + PDS_PJ1_RANGE_APO_RJ) * R_JUPITER_REF_KM / 2.0
)
ECCENTRICITY = (
    (PDS_PJ1_RANGE_APO_RJ - PDS_PJ1_RANGE_PERI_RJ)
    / (PDS_PJ1_RANGE_APO_RJ + PDS_PJ1_RANGE_PERI_RJ)
)

KM_TO_M = 1000.0
TWO_PI = 6.283185307179586

EV_JOULE = 1.602176634e-19
MEDIATOR_MASS_PREFACTOR_EV = HBAR * C / EV_JOULE
