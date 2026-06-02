"""
Jupiter mass density rho(r) for eq. (2.3).

Singh et al. JHEP01(2025)098 cite Militzer & Hubbard (2024) two-layer models.
Constant-density core + envelope (Tab. 1: M_compact = 4 M_Earth, Z2 = 0.0931).
"""

import math

from .constants import R_JUPITER

M_EARTH_KG = 5.9722e24
M_JUPITER_KG = 1.89813e27
M_CORE_KG = 4.0 * M_EARTH_KG
R_JUPITER_M = R_JUPITER
R_CORE_M = (0.0931 ** (1.0 / 3.0)) * R_JUPITER_M

_rho_core = M_CORE_KG / ((4.0 / 3.0) * math.pi * R_CORE_M**3)
M_ENVELOPE_KG = M_JUPITER_KG - M_CORE_KG
_rho_envelope = M_ENVELOPE_KG / (
    (4.0 / 3.0) * math.pi * (R_JUPITER_M**3 - R_CORE_M**3)
)


def jupiter_density(r_m):
    """Mass density [kg m^-3] at spherical radius r_m from Jupiter center."""
    if r_m < 0.0:
        raise ValueError("radius must be non-negative")
    if r_m >= R_JUPITER_M:
        return 0.0
    if r_m < R_CORE_M:
        return _rho_core
    return _rho_envelope
