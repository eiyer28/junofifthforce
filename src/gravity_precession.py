"""
Gravity-induced per-orbit precession and eq. (3.5) uncertainty propagation.

Appendix B (``app:eqs``) gives cycle-averaged drifts ``⟨Δω⟩_{J_l}`` linear in each
zonal coefficient. Because each expression is linear in ``J_l``, the gradient entry
used in eq. (3.5) is the same formula evaluated at ``J_l = 1``:

    ∂⟨Δω_g⟩/∂J_l = ⟨Δω⟩_{J_l}|_{J_l=1}

The ``μ`` entry follows from the GR piece in eq. (3.1), which scales as ``μ²``.
"""

import math
from dataclasses import dataclass
from . import constants as const
from .gravity_covariance import COVARIANCE_JHEP_ONE_SIGMA, variance_delta_omega

# Durante et al. 2020 Table 2 central values (unnormalized J_l).
MU_KM3_S2 = 126_686_534.1
J_CENTRAL = (
    0.0,  # placeholder index 0
    14696.5735e-6,
    -0.0450e-6,
    -586.6085e-6,
    -0.0723e-6,
    34.2007e-6,
    0.120e-6,
    -2.422e-6,
    -0.113e-6,
    0.181e-6,
)


@dataclass(frozen=True)
class OrbitState:
    """Keplerian elements for eqs. (3.1)–(3.4). Distances in km, angles in rad."""

    semi_major_axis_km: float
    eccentricity: float
    inclination: float
    argument_of_periapsis: float


@dataclass(frozen=True)
class GravityField:
    """JHEP parameter vector J = [μ, J₂, …, J_N]. μ in km³ s⁻², J_l dimensionless."""

    mu_km3_s2: float
    j_coefficients: tuple


def default_orbit():
    return OrbitState(
        semi_major_axis_km=const.SEMI_MAJOR_AXIS_KM,
        eccentricity=const.ECCENTRICITY,
        inclination=const.INCLINATION,
        argument_of_periapsis=const.ARGUMENT_OF_PERIAPSIS,
    )


def default_gravity_field(max_degree= const.N_HARMONICS):
    if max_degree < 2:
        return GravityField(mu_km3_s2=MU_KM3_S2, j_coefficients=tuple())
    return GravityField(
        mu_km3_s2=MU_KM3_S2,
        j_coefficients=tuple(J_CENTRAL[1:max_degree]),
    )


def _e2(e):
    return 1.0 - e * e


def _es(e):
    return e * e


def _csc(x):
    return 1.0 / math.sin(x)


# ---------------------------------------------------------------------------
# Appendix B.1 — ⟨Δω⟩_{J_l} (LaTeX app:eqs, one function per displayed equation)
# ---------------------------------------------------------------------------


def delta_omega_j2(
    j2,
    orbit,
    *,
    r_jupiter_km= const.R_JUPITER_REF_KM,
):
    """⟨Δω⟩_{J_2} from Appendix B (first displayed equation)."""
    a = orbit.semi_major_axis_km
    e2 = _e2(orbit.eccentricity)
    i = orbit.inclination
    return (
        3.0
        * math.pi
        * (3.0 + 5.0 * math.cos(2.0 * i))
        * j2
        * r_jupiter_km**2
        / (4.0 * a**2 * e2**2)
    )


def delta_omega_j3(
    j3,
    orbit,
    *,
    r_jupiter_km= const.R_JUPITER_REF_KM,
):
    """⟨Δω⟩_{J_3} from Appendix B."""
    a = orbit.semi_major_axis_km
    e = orbit.eccentricity
    e2 = _e2(e)
    i = orbit.inclination
    omega = orbit.argument_of_periapsis
    bracket = (
        -1.0
        - 3.0 * e * e
        - 4.0 * math.cos(2.0 * i)
        + 5.0 * (1.0 + 7.0 * e * e) * math.cos(4.0 * i)
    )
    return (
        3.0
        * math.pi
        * bracket
        * _csc(i)
        * math.sin(omega)
        * j3
        * r_jupiter_km**3
        / (32.0 * a**3 * e * e2**3)
    )


def delta_omega_j4(
    j4,
    orbit,
    *,
    r_jupiter_km= const.R_JUPITER_REF_KM,
):
    """⟨Δω⟩_{J_4} from Appendix B."""
    a = orbit.semi_major_axis_km
    e = orbit.eccentricity
    e2 = _e2(e)
    es = _es(e)
    i = orbit.inclination
    omega = orbit.argument_of_periapsis
    bracket = 15.0 * math.pi * (
        -27.0 * (4.0 + 5.0 * es)
        + 2.0 * (-6.0 + 5.0 * es) * math.cos(2.0 * omega)
        + 4.0
        * math.cos(2.0 * i)
        * (-52.0 - 63.0 * es + 2.0 * (-2.0 + 7.0 * es) * math.cos(2.0 * omega))
        + 7.0
        * math.cos(4.0 * i)
        * (-28.0 - 27.0 * es + 2.0 * (2.0 + 9.0 * es) * math.cos(2.0 * omega))
    )
    return bracket * j4 * r_jupiter_km**4 / (512.0 * a**4 * e2**4)


def delta_omega_j5(
    j5,
    orbit,
    *,
    r_jupiter_km= const.R_JUPITER_REF_KM,
):
    """⟨Δω⟩_{J_5} from Appendix B."""
    a = orbit.semi_major_axis_km
    e = orbit.eccentricity
    e2 = _e2(e)
    es = _es(e)
    i = orbit.inclination
    omega = orbit.argument_of_periapsis
    sin_i = math.sin(i)
    cos_i = math.cos(i)
    bracket = 15.0 * math.pi * (
        (4.0 + 41.0 * es + 18.0 * es * es)
        * (2.0 * sin_i + 7.0 * (math.sin(3.0 * i) + 3.0 * math.sin(5.0 * i)))
        * math.sin(omega)
        + 28.0 * es * (1.0 + 2.0 * es) * (7.0 + 9.0 * math.cos(2.0 * i)) * sin_i**3 * math.sin(3.0 * omega)
        - es
        * cos_i
        * (
            (4.0 + 3.0 * es)
            * (2.0 * cos_i + 21.0 * (math.cos(3.0 * i) + 5.0 * math.cos(5.0 * i)))
            * _csc(i)
            * math.sin(omega)
            + 7.0 * es * (2.0 * math.sin(2.0 * i) + 15.0 * math.sin(4.0 * i)) * math.sin(3.0 * omega)
        )
    )
    return bracket * j5 * r_jupiter_km**5 / (1024.0 * a**5 * e * e2**5)


def delta_omega_j6(
    j6,
    orbit,
    *,
    r_jupiter_km= const.R_JUPITER_REF_KM,
):
    """⟨Δω⟩_{J_6} from Appendix B."""
    a = orbit.semi_major_axis_km
    e = orbit.eccentricity
    e2 = _e2(e)
    es = _es(e)
    i = orbit.inclination
    omega = orbit.argument_of_periapsis
    sin_i = math.sin(i)
    cos2i = math.cos(2.0 * i)
    cos4i = math.cos(4.0 * i)
    cos6i = math.cos(6.0 * i)
    cos2w = math.cos(2.0 * omega)
    cos4w = math.cos(4.0 * omega)
    bracket = 105.0 * math.pi * (2256.0 * cos4i + 2376.0 * cos6i)
    bracket += 5.0 * (
        (472.0 + 1940.0 * es + 675.0 * es * es) * cos2i
        + 3.0 * es * (2.0 * (292.0 + 99.0 * es) * cos4i + 11.0 * (44.0 + 13.0 * es) * cos6i)
    )
    bracket -= 5.0 * (
        10.0 * es * (6.0 + 7.0 * es)
        + (-68.0 + 254.0 * es + 195.0 * es * es) * cos2i
        + 6.0 * (-4.0 + 102.0 * es + 55.0 * es * es) * cos4i
        + 33.0 * (4.0 + 34.0 * es + 13.0 * es * es) * cos6i
    ) * cos2w
    bracket += 50.0 * (24.0 + 100.0 * es + 35.0 * es * es + 4.0 * cos2w)
    bracket -= (
        6.0
        * es
        * (-28.0 + 45.0 * es + 4.0 * (-4.0 + 33.0 * es) * cos2i + 11.0 * (4.0 + 13.0 * es) * cos4i)
        * cos4w
        * sin_i**2
    )
    return bracket * j6 * r_jupiter_km**6 / (32768.0 * a**6 * e2**6)


def delta_omega_j7(
    j7,
    orbit,
    *,
    r_jupiter_km= const.R_JUPITER_REF_KM,
):
    """⟨Δω⟩_{J_7} from Appendix B (sin[i]^3 and csc[i] powers as in LaTeX)."""
    a = orbit.semi_major_axis_km
    e = orbit.eccentricity
    e2 = _e2(e)
    es = _es(e)
    i = orbit.inclination
    omega = orbit.argument_of_periapsis
    sin_i = math.sin(i)
    cos2i = math.cos(2.0 * i)
    cos4i = math.cos(4.0 * i)
    cos6i = math.cos(6.0 * i)
    cos8i = math.cos(8.0 * i)
    pref = 21.0 * math.pi * sin_i**3 / (524288.0 * a**7 * e * e2**7)
    t1 = -5.0 * (
        25.0 * (8.0 + 148.0 * es + 205.0 * es * es + 35.0 * es**3)
        + (448.0 + 6592.0 * es + 7240.0 * es * es + 900.0 * es**3) * cos2i
        + 12.0 * (56.0 - 5.0 * es * (-76.0 + 41.0 * es + 33.0 * es * es)) * cos4i
        - 132.0 * (-16.0 - 80.0 * es + 65.0 * es * es * (2.0 + es)) * cos6i
        - 429.0 * (8.0 + 212.0 * es + 365.0 * es * es + 75.0 * es**3) * cos8i
    ) * _csc(i) ** 4 * math.sin(omega)
    t2 = (
        30.0
        * es
        * (
            14.0 * (15.0 + 7.0 * es) * (-8.0 + 27.0 * es)
            + (-2280.0 + 13687.0 * es + 6237.0 * es * es) * cos2i
            + 22.0 * (24.0 + 967.0 * es + 351.0 * es * es) * cos4i
            + 143.0 * (24.0 + 151.0 * es + 45.0 * es * es) * cos6i
        )
        * _csc(i) ** 2
        * math.sin(3.0 * omega)
    )
    t3 = (
        264.0
        * es
        * es
        * (-45.0 + 77.0 * es + 4.0 * (-5.0 + 52.0 * es) * cos2i + 65.0 * (1.0 + 3.0 * es) * cos4i)
        * math.sin(5.0 * omega)
    )
    return pref * (t1 + t2 + t3) * j7 * r_jupiter_km**7


def delta_omega_j8(
    j8,
    orbit,
    *,
    r_jupiter_km= const.R_JUPITER_REF_KM,
):
    """⟨Δω⟩_{J_8} from Appendix B."""
    a = orbit.semi_major_axis_km
    e = orbit.eccentricity
    e2 = _e2(e)
    es = _es(e)
    i = orbit.inclination
    omega = orbit.argument_of_periapsis
    sin_i = math.sin(i)
    cos2i = math.cos(2.0 * i)
    cos4i = math.cos(4.0 * i)
    cos6i = math.cos(6.0 * i)
    cos8i = math.cos(8.0 * i)
    pref = 63.0 * math.pi / (16777216.0 * a**8 * e2**8)
    poly = 5.0 * (
        -1225.0 * (192.0 + 35.0 * es * (48.0 + 56.0 * es + 9.0 * es * es))
        - 280.0 * (1664.0 + 7.0 * es * (2064.0 + 2400.0 * es + 385.0 * es * es)) * cos2i
        - 308.0 * (1472.0 + 7.0 * es * (1776.0 + 2040.0 * es + 325.0 * es * es)) * cos4i
        - 3432.0 * (128.0 + 7.0 * es * (144.0 + 160.0 * es + 25.0 * es * es)) * cos6i
        - 715.0 * (704.0 + 7.0 * es * (624.0 + 600.0 * es + 85.0 * es * es)) * cos8i
    )
    poly += 70.0 * (
        35.0 * (-96.0 + 208.0 * es + 950.0 * es * es + 225.0 * es**3)
        + 16.0 * (-384.0 + 1648.0 * es + 5160.0 * es * es + 1155.0 * es**3) * cos2i
        + 44.0 * (-96.0 + 1360.0 * es + 2870.0 * es * es + 585.0 * es**3) * cos4i
        + 2288.0 * es * (48.0 + 80.0 * es + 15.0 * es * es) * cos6i
        + 143.0 * (96.0 + 1328.0 * es + 1610.0 * es * es + 255.0 * es**3) * cos8i
    ) * math.cos(2.0 * omega)
    poly += (
        616.0
        * es
        * (
            6.0 * (-280.0 + 944.0 * es + 363.0 * es * es)
            + (-1960.0 + 14128.0 * es + 4797.0 * es * es) * cos2i
            + 26.0 * (40.0 + 688.0 * es + 195.0 * es * es) * cos4i
            + 65.0 * (40.0 + 208.0 * es + 51.0 * es * es) * cos6i
        )
        * math.cos(4.0 * omega)
        * sin_i**2
    )
    poly += (
        4576.0
        * es
        * es
        * (-22.0 + 39.0 * es + 4.0 * (-2.0 + 25.0 * es) * cos2i + 5.0 * (6.0 + 17.0 * es) * cos4i)
        * math.cos(6.0 * omega)
        * sin_i**4
    )
    return pref * poly * j8 * r_jupiter_km**8


def delta_omega_j9(
    j9,
    orbit,
    *,
    r_jupiter_km= const.R_JUPITER_REF_KM,
):
    """⟨Δω⟩_{J_9} from Appendix B (sin[i]^5 and csc[i] powers as in LaTeX)."""
    a = orbit.semi_major_axis_km
    e = orbit.eccentricity
    e2 = _e2(e)
    es = _es(e)
    i = orbit.inclination
    omega = orbit.argument_of_periapsis
    sin_i = math.sin(i)
    cos2i = math.cos(2.0 * i)
    cos4i = math.cos(4.0 * i)
    cos6i = math.cos(6.0 * i)
    cos8i = math.cos(8.0 * i)
    cos10i = math.cos(10.0 * i)
    pref = 45.0 * math.pi * sin_i**5 / (67108864.0 * a**9 * e * e2**9)
    t1 = 7.0 * (
        98.0 * (64.0 + 1968.0 * es + 5768.0 * es * es + 3325.0 * es**3 + 315.0 * es**4)
        + 14.0
        * (960.0 + 25808.0 * es + 67032.0 * es * es + 33635.0 * es**3 + 2695.0 * es**4)
        * cos2i
        - 88.0 * (-192.0 - 3344.0 * es - 3864.0 * es * es + 1225.0 * es**3 + 455.0 * es**4) * cos4i
        - 429.0 * (-64.0 + 7.0 * es * (-80.0 + 232.0 * es + 405.0 * es * es + 65.0 * es**3)) * cos6i
        - 286.0 * (-320.0 + 7.0 * es * (-528.0 + 488.0 * es + 1465.0 * es * es + 255.0 * es**3)) * cos8i
        - 2431.0 * (64.0 + 2608.0 * es + 9128.0 * es * es + 6125.0 * es**3 + 665.0 * es**4) * cos10i
    ) * _csc(i) ** 6 * math.sin(omega)
    t2 = (
        -4312.0
        * es
        * (
            21.0 * (-80.0 + 372.0 * es + 705.0 * es * es + 121.0 * es**3)
            + 28.0 * (-96.0 + 712.0 * es + 1178.0 * es * es + 195.0 * es**3) * cos2i
            + 52.0 * (-16.0 + 580.0 * es + 773.0 * es * es + 119.0 * es**3) * cos4i
            + 52.0 * (32.0 + 744.0 * es + 834.0 * es * es + 119.0 * es**3) * cos6i
            + 221.0 * (16.0 + 156.0 * es + 147.0 * es * es + 19.0 * es**3) * cos8i
        )
        * _csc(i) ** 4
        * math.sin(3.0 * omega)
    )
    t3 = (
        -32032.0
        * es
        * es
        * (
            22.0 * (-12.0 + 43.0 * es + 13.0 * es * es)
            + (-276.0 + 2213.0 * es + 605.0 * es * es) * cos2i
            + (200.0 + 2462.0 * es + 578.0 * es * es) * cos4i
            + 17.0 * (20.0 + 91.0 * es + 19.0 * es * es) * cos6i
        )
        * _csc(i) ** 2
        * math.sin(5.0 * omega)
    )
    t4 = (
        -4576.0
        * es**3
        * (-91.0 + 165.0 * es + 4.0 * (-7.0 + 102.0 * es) * cos2i + 17.0 * (7.0 + 19.0 * es) * cos4i)
        * math.sin(7.0 * omega)
    )
    return pref * (t1 + t2 + t3 + t4) * j9 * r_jupiter_km**9


def delta_omega_j10(
    j10,
    orbit,
    *,
    r_jupiter_km= const.R_JUPITER_REF_KM,
):
    """⟨Δω⟩_{J_{10}} from Appendix B."""
    a = orbit.semi_major_axis_km
    e = orbit.eccentricity
    e2 = _e2(e)
    es = _es(e)
    i = orbit.inclination
    omega = orbit.argument_of_periapsis
    sin_i = math.sin(i)
    cos2i = math.cos(2.0 * i)
    cos4i = math.cos(4.0 * i)
    cos6i = math.cos(6.0 * i)
    cos8i = math.cos(8.0 * i)
    cos10i = math.cos(10.0 * i)
    pref = 495.0 * math.pi / (2147483648.0 * a**10 * e2**10)
    poly = 7.0 * (
        23814.0 * (128.0 + 3.0 * es * (640.0 + 1568.0 * es + 840.0 * es * es + 77.0 * es**3))
        + 294.0
        * (20608.0 + 307584.0 * es + 751968.0 * es * es + 402360.0 * es**3 + 36855.0 * es**4)
        * cos2i
        + 312.0
        * (19072.0 + 279936.0 * es + 679392.0 * es * es + 362040.0 * es**3 + 33075.0 * es**4)
        * cos4i
        + 1053.0
        * (5504.0 + 21.0 * es * (3712.0 + 8864.0 * es + 4680.0 * es * es + 425.0 * es**3))
        * cos6i
        + 442.0
        * (12928.0 + 21.0 * es * (8064.0 + 18528.0 * es + 9560.0 * es * es + 855.0 * es**3))
        * cos8i
        + 4199.0
        * (1664.0 + 3.0 * es * (5760.0 + 7.0 * es * (1632.0 + 760.0 * es + 63.0 * es * es)))
        * cos10i
    )
    poly -= 84.0 * (
        294.0 * (-192.0 + 544.0 * es + 6020.0 * es * es + 4802.0 * es**3 + 539.0 * es**4)
        + 98.0
        * (-1088.0 + 5344.0 * es + 42028.0 * es * es + 32158.0 * es**3 + 3549.0 * es**4)
        * cos2i
        + 104.0
        * (-832.0 + 10720.0 * es + 55356.0 * es * es + 39102.0 * es**3 + 4165.0 * es**4)
        * cos4i
        + 13.0
        * (-3648.0 + 156640.0 * es + 626444.0 * es * es + 411278.0 * es**3 + 42245.0 * es**4)
        * cos6i
        + 442.0 * (64.0 + 7328.0 * es + 24276.0 * es * es + 14826.0 * es**3 + 1463.0 * es**4) * cos8i
        + 4199.0 * (64.0 + 1312.0 * es + 3220.0 * es * es + 1666.0 * es**3 + 147.0 * es**4) * cos10i
    ) * math.cos(2.0 * omega)
    poly -= 8736.0 * es * (
        33.0 * (-224.0 + 1208.0 * es + 1600.0 * es * es + 221.0 * es**3)
        + 24.0 * (-448.0 + 3896.0 * es + 4680.0 * es * es + 627.0 * es**3) * cos2i
        + 4.0 * (-224.0 + 30040.0 * es + 30432.0 * es * es + 3825.0 * es**3) * cos4i
        + 136.0 * (64.0 + 920.0 * es + 808.0 * es * es + 95.0 * es**3) * cos6i
        + 323.0 * (32.0 + 248.0 * es + 192.0 * es * es + 21.0 * es**3) * cos8i
    ) * math.cos(4.0 * omega) * sin_i**2
    poly -= 1248.0 * es * es * (
        26.0 * (-924.0 + 3418.0 * es + 855.0 * es * es)
        + (-23100.0 + 199186.0 * es + 45747.0 * es * es) * cos2i
        + 238.0 * (84.0 + 850.0 * es + 171.0 * es * es) * cos4i
        + 323.0 * (84.0 + 346.0 * es + 63.0 * es * es) * cos6i
    ) * math.cos(6.0 * omega) * sin_i**4
    poly -= 7072.0 * es**3 * (
        -120.0 + 221.0 * es + 4.0 * (-8.0 + 133.0 * es) * cos2i + 19.0 * (8.0 + 21.0 * es) * cos4i
    ) * math.cos(8.0 * omega) * sin_i**6
    return pref * poly * j10 * r_jupiter_km**10


_APPENDIX_B = {
    2: delta_omega_j2,
    3: delta_omega_j3,
    4: delta_omega_j4,
    5: delta_omega_j5,
    6: delta_omega_j6,
    7: delta_omega_j7,
    8: delta_omega_j8,
    9: delta_omega_j9,
    10: delta_omega_j10,
}


def delta_omega_jl_analytic(
    degree,
    j_l,
    orbit,
    *,
    r_jupiter_km= const.R_JUPITER_REF_KM,
):
    """Dispatch to the Appendix B formula for ``⟨Δω⟩_{J_l}`` [rad orbit⁻¹]."""
    try:
        formula = _APPENDIX_B[degree]
    except KeyError as exc:
        raise ValueError(f"Appendix B precession formula not implemented for J_{degree}") from exc
    return formula(j_l, orbit, r_jupiter_km=r_jupiter_km)


def mean_precession_harmonics_analytic(
    orbit=None,
    field=None,
    *,
    max_degree= const.N_HARMONICS,
    r_jupiter_km= const.R_JUPITER_REF_KM,
):
    """Sum of Appendix B zonal-harmonic contributions at the supplied coefficients."""
    orbit = orbit or default_orbit()
    field = field or default_gravity_field(max_degree)
    total = 0.0
    for degree, j_l in enumerate(field.j_coefficients, start=2):
        total += delta_omega_jl_analytic(degree, j_l, orbit, r_jupiter_km=r_jupiter_km)
    return total


# ---------------------------------------------------------------------------
# GR piece (eq. 3.4) — only source of ∂⟨Δω_g⟩/∂μ
# ---------------------------------------------------------------------------


def _radius(true_anomaly, orbit):
    a, e = orbit.semi_major_axis_km, orbit.eccentricity
    return a * (1.0 - e * e) / (1.0 + e * math.cos(true_anomaly))


def _mean_precession_gr_integral(
    orbit,
    mu_km3_s2,
    *,
    n_steps= 4_096,
    r_jupiter_km= const.R_JUPITER_REF_KM,
):
    """GR-only ⟨Δω_g⟩ from eq. (3.4), with all J_l = 0 [rad orbit⁻¹]."""
    field = GravityField(mu_km3_s2=mu_km3_s2, j_coefficients=tuple())
    a, e = orbit.semi_major_axis_km, orbit.eccentricity
    i, omega = orbit.inclination, orbit.argument_of_periapsis
    c_km_s = const.C / const.KM_TO_M
    h = math.sqrt(mu_km3_s2 * a * (1.0 - e * e))
    df = const.TWO_PI / n_steps
    total = 0.0
    for step in range(n_steps):
        f = (step + 0.5) * df
        r = _radius(f, orbit)
        gr_coeff = mu_km3_s2 * a * (1.0 - e * e) / (c_km_s * c_km_s)
        bracket = 1.0 + gr_coeff / r
        dbr_dr = -gr_coeff / (r * r)
        dV_dr = -mu_km3_s2 / (r * r) * bracket - (mu_km3_s2 / r) * dbr_dr
        dV_dz = 0.0
        sin_i = math.sin(i)
        sin_wf = math.sin(omega + f)
        cos_wf = math.cos(omega + f)
        f_r = -dV_dr + sin_i * sin_wf * dV_dz
        f_t = sin_i * cos_wf * dV_dz
        f_n = math.cos(i) * dV_dz
        prefactor = -math.sqrt(a * (1.0 - e * e)) / (
            e * mu_km3_s2 * (1.0 + math.cos(f)) * sin_i
        )
        omega_dot = prefactor * (
            math.cos(f) * f_r
            + (2.0 + math.cos(f)) * math.sin(f) / e * f_t
            - math.sin(omega + f) * f_n
        )
        total += (omega_dot * r / h) * df
    return total / const.TWO_PI


# ---------------------------------------------------------------------------
# Eq. (3.5) gradient g = [∂⟨Δω_g⟩/∂μ, ∂⟨Δω_g⟩/∂J₂, …]
# ---------------------------------------------------------------------------


def gradient_jl_coefficient(
    degree,
    orbit=None,
    *,
    r_jupiter_km= const.R_JUPITER_REF_KM,
):
    """∂⟨Δω_g⟩/∂J_l = ⟨Δω⟩_{J_l} with J_l = 1 (Appendix B is linear in J_l)."""
    orbit = orbit or default_orbit()
    return delta_omega_jl_analytic(degree, 1.0, orbit, r_jupiter_km=r_jupiter_km)


def gradient_wrt_mu(
    orbit=None,
    field=None,
    *,
    n_steps= 4_096,
    r_jupiter_km= const.R_JUPITER_REF_KM,
):
    """∂⟨Δω_g⟩/∂μ = 2 ⟨Δω_g⟩_GR / μ (harmonic terms independent of μ)."""
    orbit = orbit or default_orbit()
    field = field or default_gravity_field()
    mu = field.mu_km3_s2
    if mu == 0.0:
        raise ValueError("mu must be non-zero")
    gr = _mean_precession_gr_integral(orbit, mu, n_steps=n_steps, r_jupiter_km=r_jupiter_km)
    return 2.0 * gr / mu


def precession_gradient(
    orbit=None,
    field=None,
    *,
    max_degree= const.N_HARMONICS,
    n_steps= 4_096,
    r_jupiter_km= const.R_JUPITER_REF_KM,
):
    """Gradient vector for eq. (3.5): Appendix B formulas at J_l=1, plus ∂/∂μ."""
    orbit = orbit or default_orbit()
    field = field or default_gravity_field(max_degree)
    grad_mu = gradient_wrt_mu(orbit, field, n_steps=n_steps, r_jupiter_km=r_jupiter_km)
    grad_j = tuple(
        gradient_jl_coefficient(degree, orbit, r_jupiter_km=r_jupiter_km)
        for degree in range(2, max_degree + 1)
    )
    return (grad_mu, *grad_j)


def sigma_omega(
    orbit=None,
    field=None,
    *,
    max_degree= const.N_HARMONICS,
    n_steps= 4_096,
    covariance=None,
    r_jupiter_km= const.R_JUPITER_REF_KM,
):
    """σ_ω from eq. (3.5) [rad orbit⁻¹]."""
    gradient = precession_gradient(
        orbit,
        field,
        max_degree=max_degree,
        n_steps=n_steps,
        r_jupiter_km=r_jupiter_km,
    )
    return math.sqrt(
        variance_delta_omega(gradient, covariance or COVARIANCE_JHEP_ONE_SIGMA)
    )


if __name__ == "__main__":
    from .gravity_covariance import PARAMETER_LABELS

    orbit = default_orbit()
    field = default_gravity_field()
    gradient = precession_gradient(orbit, field)
    sigma2 = variance_delta_omega(gradient)
    sigma = math.sqrt(sigma2)
    harmonics = mean_precession_harmonics_analytic(orbit, field)
    gr_only = _mean_precession_gr_integral(orbit, field.mu_km3_s2)

    print("gravity_precession  (JHEP eq. 3.5 execution)")
    print()
    print("Orbital parameters used for the analysis:")
    print(f"  a     = {orbit.semi_major_axis_km:.3f} km")
    print(f"  e     = {orbit.eccentricity:.6f}")
    print(f"  i     = {orbit.inclination:.4f} rad")
    print(f"  omega = {orbit.argument_of_periapsis:.4f} rad")
    print(f"  R_X   = {const.R_JUPITER_REF_KM:.1f} km")
    print()
    print("Mean precession <dw_g> [rad/orbit]:")
    print(f"  zonal harmonics (Appendix B) = {harmonics:.6e}")
    print(f"  GR only (eq. 3.4 integral)     = {gr_only:.6e}")
    print()
    print("Gradient g = d<dw_g>/dJ  (for sigma_omega^2 = g^T C_J g):")
    for label, value in zip(PARAMETER_LABELS, gradient):
        if label == "GM":
            unit = "rad orbit^-1 / (km^3 s^-2)"
        else:
            unit = "rad orbit^-1"
        print(f"  {label:8s}  {value: .6e}  [{unit}]")
    print()
    print("Uncertainty from Durante C_J (1-sigma):")
    print(f"  sigma_omega^2 = {sigma2:.6e}  (rad/orbit)^2")
    print(f"  sigma_omega   = {sigma:.6e}  rad/orbit")
    print()
    print("Validation against Singh et al. (N=10, i=1.57, omega=3.08):")
    print(f"  sigma_omega target  {const.SIGMA_OMEGA_TARGET:.1e} rad/orbit")
    rel = abs(sigma - const.SIGMA_OMEGA_TARGET) / const.SIGMA_OMEGA_TARGET
    print(f"  relative error      {rel:.1%}")
    print()
