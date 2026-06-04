"""
Fifth-force orbital precession and JHEP Section 4 constraints.

Yukawa potential eq. (2.1)–(2.3), Gauss equation (2.4), orbit average (2.5).
"""

import math

import numpy as np

from . import constants as const
from .gravity_precession import MU_KM3_S2, default_orbit
from .jupiter_density import R_JUPITER_M, jupiter_density

G_SI = const.G
# Eq. (2.4) uses a, μ, and r(f) in km but ∂U/∂r from the SI volume integral (eq. 2.3) is per metre.
# Do not rescale ∂U/∂r by KM_TO_M — that factor of 1000 inflates ⟨Δω⟩ by ~10³.
LAMBDA_REFERENCE_INF = 1e20
# Subtract the λ→∞ (Newtonian-Yukawa) radial derivative only for λ ≳ R_X. At smaller λ the Yukawa
# integral is already screened; subtracting the large-λ limit spuriously blows up the small-λ edge.
LAMBDA_REFERENCE_THRESHOLD_M = const.R_JUPITER


def _orbit_radius_km(true_anomaly, orbit):
    a_km = orbit.semi_major_axis_km
    e = orbit.eccentricity
    return a_km * (1.0 - e * e) / (1.0 + e * math.cos(true_anomaly))


def _orbit_radius_m(true_anomaly, orbit):
    return _orbit_radius_km(true_anomaly, orbit) * const.KM_TO_M


def _yukawa_kernel(R_m, r_prime_m, theta, lam_m):
    cos_t = math.cos(theta)
    dist = math.sqrt(R_m * R_m + r_prime_m * r_prime_m - 2.0 * R_m * r_prime_m * cos_t)
    if dist < 1e-12:
        dist = 1e-12
    return dist, math.exp(-dist / lam_m)


def specific_potential_at_R(R_m, lam_m, alpha=1.0, *, n_r=80, n_theta=48):
    """Specific fifth-force potential U/m_Juno at Juno [m^2 s^-2], eq. (2.3)."""
    if lam_m <= 0.0:
        raise ValueError("lambda must be positive")
    if R_m <= 0.0:
        raise ValueError("R must be positive")

    total = 0.0
    r_grid = np.linspace(0.0, R_JUPITER_M, n_r)
    t_grid = np.linspace(0.0, math.pi, n_theta)
    dr = r_grid[1] - r_grid[0] if n_r > 1 else R_JUPITER_M
    dt = t_grid[1] - t_grid[0] if n_theta > 1 else math.pi

    for r_prime in r_grid[1:-1]:
        rho = jupiter_density(float(r_prime))
        if rho <= 0.0:
            continue
        for theta in t_grid[1:-1]:
            dist, yuk = _yukawa_kernel(R_m, float(r_prime), float(theta), lam_m)
            total += (
                alpha
                * G_SI
                * rho
                * r_prime
                * r_prime
                * math.sin(theta)
                * yuk
                / dist
                * dr
                * dt
            )
    return -2.0 * math.pi * total


def dU_dR(R_m, lam_m, alpha=1.0, *, rel_step=1e-4, subtract_reference=True):
    """
    Radial derivative dU/dR [m s^-2] for eq. (2.4).

    When subtract_reference is True, use ∂U/∂r|_λ − ∂U/∂r|_{λ→∞} so the signal
  falls off at large λ (Figure 2 right panel; α_max ∝ λ² in Figure 4).
    """
    step = max(R_m * rel_step, 1.0)

    def _dU_at(lam):
        u_plus = specific_potential_at_R(R_m + step, lam, alpha=alpha)
        u_minus = specific_potential_at_R(max(R_m - step, step), lam, alpha=alpha)
        return (u_plus - u_minus) / (2.0 * step)

    dudr = _dU_at(lam_m)
    if (
        subtract_reference
        and lam_m >= LAMBDA_REFERENCE_THRESHOLD_M
        and lam_m < LAMBDA_REFERENCE_INF / 10.0
    ):
        dudr -= _dU_at(LAMBDA_REFERENCE_INF)
    return dudr


def mean_fifth_force_precession(
    alpha,
    lam_m,
    orbit=None,
    *,
    n_steps=4096,
    n_r=80,
    n_theta=48,
):
    """Orbit-averaged ⟨Δω⟩ [rad orbit^-1], eqs. (2.4)–(2.5). Linear in alpha."""
    orbit = orbit or default_orbit()
    a_km = orbit.semi_major_axis_km
    e = orbit.eccentricity
    sin_i = math.sin(orbit.inclination)
    if abs(sin_i) < 1e-15:
        raise ValueError("inclination too close to 0 for eq. (2.4)")

    p_km = a_km * (1.0 - e * e)
    h_km = math.sqrt(MU_KM3_S2 * p_km)
    df = const.TWO_PI / n_steps
    # Eq. (2.4): a, μ in km; ∂U/∂r from eq. (2.3) in SI (per metre).
    pref_base = -math.sqrt(p_km) / (e * e * MU_KM3_S2)
    total = 0.0

    for step in range(n_steps):
        f = (step + 0.5) * df
        r_km = _orbit_radius_km(f, orbit)
        r_m = r_km * const.KM_TO_M
        dudr = dU_dR(r_m, lam_m, alpha=alpha, rel_step=1e-4, subtract_reference=True)
        omega_dot = pref_base * dudr * math.cos(f)
        total += omega_dot * (r_km * r_km / h_km) * df

    return total / const.TWO_PI


def mean_fifth_force_precession_unit(lam_m, orbit=None, **kwargs):
    """⟨Δω⟩ at alpha = 1."""
    return mean_fifth_force_precession(1.0, lam_m, orbit=orbit, **kwargs)


def alpha_upper_bound(lam_m, sigma_omega, orbit=None, *, confidence_factor=None, **kwargs):
    """95% C.L. upper bound on |alpha| from |⟨Δω⟩| <= factor * sigma_omega."""
    factor = confidence_factor if confidence_factor is not None else const.CONFIDENCE_SIGMA_FACTOR
    limit = factor * sigma_omega
    dw1 = mean_fifth_force_precession_unit(lam_m, orbit=orbit, **kwargs)
    if abs(dw1) < 1e-30:
        return float("nan")
    return limit / abs(dw1)


def constraint_curve(lambdas_m, sigma_omega, orbit=None, *, confidence_factor=None, **kwargs):
    orbit = orbit or default_orbit()
    alphas = [
        alpha_upper_bound(
            float(lam), sigma_omega, orbit=orbit, confidence_factor=confidence_factor, **kwargs
        )
        for lam in lambdas_m
    ]
    return np.asarray(lambdas_m, dtype=float), np.asarray(alphas, dtype=float)
