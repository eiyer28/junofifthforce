"""Physical constants and Juno/Jupiter orbital parameters.

All gravity-field / sigma_omega calculations are done in kilometre units to be
consistent with the Durante covariance matrix, whose GM entry is in km^3/s^2.
The fifth-force calculation works in SI (metres, kilograms) because the force
range lambda is naturally expressed in metres.

References:
  Singh et al., JHEP 01 (2025) 098  [arXiv:2409.10616]
  Durante et al., Geophys. Res. Lett. 47 (2020) e86572  (gravity field + covariance)
"""

import numpy as np

# --- Fundamental constants -------------------------------------------------
G_SI = 6.67430e-11          # m^3 kg^-1 s^-2
C_KM_S = 299_792.458        # speed of light, km/s
C_M_S = C_KM_S * 1.0e3      # speed of light, m/s
EV_PER_KG = 1.0 / 1.782_661_92e-36   # 1 kg expressed in eV/c^2
HBAR_EV_S = 6.582_119_569e-16        # reduced Planck constant, eV*s

# --- Jupiter ---------------------------------------------------------------
# GM of Jupiter (Durante et al. 2020 system value), km^3/s^2.
MU_KM = 1.266_865_341e8     # km^3/s^2
MU_M = MU_KM * 1.0e9        # m^3/s^2  (1 km^3 = 1e9 m^3)
M_JUP_KG = MU_M / G_SI      # Jupiter mass, kg

# Reference (equatorial) radius used in the harmonic expansion, km.
R_J_KM = 71_492.0
R_J_M = R_J_KM * 1.0e3

# --- Juno orbit (PJ01-PJ17 capture orbit, ~53.5 day period) ----------------
# The orbital period of Juno's science/capture orbit used by Durante et al.
T_ORBIT_S = 53.5 * 86_400.0

# Semi-major axis from Kepler's third law: a^3 = mu * T^2 / (4 pi^2).
A_KM = (MU_KM * T_ORBIT_S**2 / (4.0 * np.pi**2)) ** (1.0 / 3.0)
A_M = A_KM * 1.0e3

# Eccentricity from the perijove radius. Juno's closest approach is ~0.05 R_J
# above the surface (Singh et al.), i.e. r_p ~ 1.06 R_J, with r_p = a (1 - e).
R_PERIJOVE_KM = 1.06 * R_J_KM
ECC = 1.0 - R_PERIJOVE_KM / A_KM

# Inclination and argument of perijove of the FIRST Juno orbit. This is the
# conservative choice (largest sigma_omega) used by Singh et al. (Fig. 3).
INC = 1.57       # rad
OMEGA = 3.08     # rad

# Highest zonal-harmonic order retained in the analysis (Singh et al. use N=10).
N_MAX = 10


def mass_to_meV_range(lam_m):
    """Convert a Yukawa range lambda [m] to a mediator mass m* [eV].

    m* = hbar c / lambda.
    """
    return HBAR_EV_S * C_M_S / np.asarray(lam_m, dtype=float)


if __name__ == "__main__":
    print(f"GM (mu)        = {MU_KM:.6e} km^3/s^2")
    print(f"Jupiter mass   = {M_JUP_KG:.6e} kg")
    print(f"R_J            = {R_J_KM:.1f} km")
    print(f"orbit period   = {T_ORBIT_S/86400:.2f} days")
    print(f"semi-major a   = {A_KM:.6e} km  ({A_KM/R_J_KM:.2f} R_J)")
    print(f"eccentricity e = {ECC:.6f}  (1 - e = {1-ECC:.5f})")
    print(f"perijove r_p   = {R_PERIJOVE_KM:.1f} km  ({R_PERIJOVE_KM/R_J_KM:.3f} R_J)")
    print(f"apojove  r_a   = {A_KM*(1+ECC):.6e} km  ({A_KM*(1+ECC)/R_J_KM:.2f} R_J)")
