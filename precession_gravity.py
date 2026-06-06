"""Gravity-induced precession of Juno and its derivatives w.r.t. the gravity-field
parameters.

This implements the cycle-averaged precession-angle drift <Delta omega_g> from the
zonal harmonics J2..J10 (Singh et al. JHEP 01 (2025) 098, Appendix B, eqs. B.1-B.9)
plus the leading general-relativistic term (proportional to mu = GM).

Each Appendix-B expression is *linear* in its zonal coefficient J_l, so the partial
derivative d<Delta omega_g>/dJ_l is simply the expression with the J_l factor removed
(keeping the R_J^l factor). These derivatives, contracted with the Durante covariance
matrix, give sigma_omega (eq. 3.5).

All lengths here are in kilometres and mu in km^3/s^2, consistent with the covariance
matrix. The derivatives w.r.t. J_l are dimensionless; the derivative w.r.t. mu has
units of (km^3/s^2)^-1.
"""

import numpy as np

from constants import C_KM_S


# Normalisation factor sqrt(2l+1): converts an unnormalised-J_l derivative to a
# fully-normalised-Jbar_l derivative, since J_l = Jbar_l * sqrt(2l+1).
def norm_factor(l):
    return np.sqrt(2 * l + 1)


def _trig(inc, omega):
    """Pre-compute the trig multiples that appear in the Appendix-B expressions."""
    return {
        "cos_i": np.cos(inc),
        "sin_i": np.sin(inc),
        "csc_i": 1.0 / np.sin(inc),
        "cos3i": np.cos(3 * inc),
        "cos5i": np.cos(5 * inc),
        "cos2i": np.cos(2 * inc),
        "cos4i": np.cos(4 * inc),
        "cos6i": np.cos(6 * inc),
        "cos8i": np.cos(8 * inc),
        "cos10i": np.cos(10 * inc),
        "sin2i": np.sin(2 * inc),
        "sin3i": np.sin(3 * inc),
        "sin4i": np.sin(4 * inc),
        "sin5i": np.sin(5 * inc),
        "sin_w": np.sin(omega),
        "sin3w": np.sin(3 * omega),
        "sin5w": np.sin(5 * omega),
        "sin7w": np.sin(7 * omega),
        "cos2w": np.cos(2 * omega),
        "cos4w": np.cos(4 * omega),
        "cos6w": np.cos(6 * omega),
        "cos8w": np.cos(8 * omega),
    }


def domega_dJl(l, a, e, inc, omega, RJ):
    """d<Delta omega_g>/dJ_l for the (unnormalised) zonal coefficient J_l, l=2..10.

    Returns the coefficient multiplying J_l in eqs. B.1-B.9 (i.e. the analytic
    partial derivative). Lengths a, RJ in the same units; result dimensionless.
    """
    e2, e4, e6, e8 = e**2, e**4, e**6, e**8
    em1 = e**2 - 1.0          # (-1 + e^2); sign preserved for odd powers
    t = _trig(inc, omega)
    cos_i, sin_i, csc_i = t["cos_i"], t["sin_i"], t["csc_i"]
    cos3i, cos5i = t["cos3i"], t["cos5i"]
    cos2i, cos4i, cos6i, cos8i, cos10i = (
        t["cos2i"], t["cos4i"], t["cos6i"], t["cos8i"], t["cos10i"])
    sin2i, sin3i, sin4i, sin5i = t["sin2i"], t["sin3i"], t["sin4i"], t["sin5i"]
    sin_w, sin3w, sin5w, sin7w = t["sin_w"], t["sin3w"], t["sin5w"], t["sin7w"]
    cos2w, cos4w, cos6w, cos8w = t["cos2w"], t["cos4w"], t["cos6w"], t["cos8w"]
    pi = np.pi

    if l == 2:  # B.1
        return 3 * pi * (3 + 5 * cos2i) * RJ**2 / (4 * a**2 * em1**2)

    if l == 3:  # B.2
        return (3 * pi * (-1 - 3 * e2 - 4 * cos2i + 5 * (1 + 7 * e2) * cos4i)
                * csc_i * sin_w * RJ**3 / (32 * a**3 * e * em1**3))

    if l == 4:  # B.3
        return (1.0 / (512 * a**4 * em1**4)) * 15 * pi * (
            -27 * (4 + 5 * e2)
            + 2 * (-6 + 5 * e2) * cos2w
            + 4 * cos2i * (-52 - 63 * e2 + 2 * (-2 + 7 * e2) * cos2w)
            + 7 * cos4i * (-28 - 27 * e2 + 2 * (2 + 9 * e2) * cos2w)
        ) * RJ**4

    if l == 5:  # B.4
        return (1.0 / (1024 * a**5 * e * em1**5)) * 15 * pi * (
            (4 + 41 * e2 + 18 * e4) * (2 * sin_i + 7 * (sin3i + 3 * sin5i)) * sin_w
            + 28 * e2 * (1 + 2 * e2) * (7 + 9 * cos2i) * sin_i**3 * sin3w
            - e2 * cos_i * (
                (4 + 3 * e2) * (2 * cos_i + 21 * (cos3i + 5 * cos5i)) * csc_i * sin_w
                + 7 * e2 * (2 * sin2i + 15 * sin4i) * sin3w
            )
        ) * RJ**5

    if l == 6:  # B.5
        return (1.0 / (32768 * a**6 * em1**6)) * 105 * pi * (
            2256 * cos4i + 2376 * cos6i
            + 5 * ((472 + 1940 * e2 + 675 * e4) * cos2i
                   + 3 * e2 * (2 * (292 + 99 * e2) * cos4i + 11 * (44 + 13 * e2) * cos6i))
            - 5 * (10 * e2 * (6 + 7 * e2)
                   + (-68 + 254 * e2 + 195 * e4) * cos2i
                   + 6 * (-4 + 102 * e2 + 55 * e4) * cos4i
                   + 33 * (4 + 34 * e2 + 13 * e4) * cos6i) * cos2w
            + 50 * (24 + 100 * e2 + 35 * e4 + 4 * cos2w)
            - 6 * e2 * (-28 + 45 * e2 + 4 * (-4 + 33 * e2) * cos2i
                        + 11 * (4 + 13 * e2) * cos4i) * cos4w * sin_i**2
        ) * RJ**6

    if l == 7:  # B.6
        return (1.0 / (524288 * a**7 * e * em1**7)) * 21 * pi * sin_i**3 * (
            -5 * (25 * (8 + 148 * e2 + 205 * e4 + 35 * e6)
                  + (448 + 6592 * e2 + 7240 * e4 + 900 * e6) * cos2i
                  + 12 * (56 - 5 * e2 * (-76 + 41 * e2 + 33 * e4)) * cos4i
                  - 132 * (-16 - 80 * e2 + 65 * e4 * (2 + e2)) * cos6i
                  - 429 * (8 + 212 * e2 + 365 * e4 + 75 * e6) * cos8i) * csc_i**4 * sin_w
            + 30 * e2 * (14 * (15 + 7 * e2) * (-8 + 27 * e2)
                         + (-2280 + 13687 * e2 + 6237 * e4) * cos2i
                         + 22 * (24 + 967 * e2 + 351 * e4) * cos4i
                         + 143 * (24 + 151 * e2 + 45 * e4) * cos6i) * csc_i**2 * sin3w
            + 264 * e4 * (-45 + 77 * e2 + 4 * (-5 + 52 * e2) * cos2i
                          + 65 * (1 + 3 * e2) * cos4i) * sin5w
        ) * RJ**7

    if l == 8:  # B.7
        return (1.0 / (16777216 * a**8 * em1**8)) * 63 * pi * (
            5 * (-1225 * (192 + 35 * e2 * (48 + 56 * e2 + 9 * e4))
                 - 280 * (1664 + 7 * e2 * (2064 + 2400 * e2 + 385 * e4)) * cos2i
                 - 308 * (1472 + 7 * e2 * (1776 + 2040 * e2 + 325 * e4)) * cos4i
                 - 3432 * (128 + 7 * e2 * (144 + 160 * e2 + 25 * e4)) * cos6i
                 - 715 * (704 + 7 * e2 * (624 + 600 * e2 + 85 * e4)) * cos8i)
            + 70 * (35 * (-96 + 208 * e2 + 950 * e4 + 225 * e6)
                    + 16 * (-384 + 1648 * e2 + 5160 * e4 + 1155 * e6) * cos2i
                    + 44 * (-96 + 1360 * e2 + 2870 * e4 + 585 * e6) * cos4i
                    + 2288 * e2 * (48 + 80 * e2 + 15 * e4) * cos6i
                    + 143 * (96 + 1328 * e2 + 1610 * e4 + 255 * e6) * cos8i) * cos2w
            + 616 * e2 * (6 * (-280 + 944 * e2 + 363 * e4)
                          + (-1960 + 14128 * e2 + 4797 * e4) * cos2i
                          + 26 * (40 + 688 * e2 + 195 * e4) * cos4i
                          + 65 * (40 + 208 * e2 + 51 * e4) * cos6i) * cos4w * sin_i**2
            + 4576 * e4 * (-22 + 39 * e2 + 4 * (-2 + 25 * e2) * cos2i
                           + 5 * (6 + 17 * e2) * cos4i) * cos6w * sin_i**4
        ) * RJ**8

    if l == 9:  # B.8
        return (1.0 / (67108864 * a**9 * e * em1**9)) * 45 * pi * sin_i**5 * (
            7 * (98 * (64 + 1968 * e2 + 5768 * e4 + 3325 * e6 + 315 * e8)
                 + 14 * (960 + 25808 * e2 + 67032 * e4 + 33635 * e6 + 2695 * e8) * cos2i
                 - 88 * (-192 - 3344 * e2 - 3864 * e4 + 1225 * e6 + 455 * e8) * cos4i
                 - 429 * (-64 + 7 * e2 * (-80 + 232 * e2 + 405 * e4 + 65 * e6)) * cos6i
                 - 286 * (-320 + 7 * e2 * (-528 + 488 * e2 + 1465 * e4 + 255 * e6)) * cos8i
                 - 2431 * (64 + 2608 * e2 + 9128 * e4 + 6125 * e6 + 665 * e8) * cos10i) * csc_i**6 * sin_w
            - 4312 * e2 * (21 * (-80 + 372 * e2 + 705 * e4 + 121 * e6)
                           + 28 * (-96 + 712 * e2 + 1178 * e4 + 195 * e6) * cos2i
                           + 52 * (-16 + 580 * e2 + 773 * e4 + 119 * e6) * cos4i
                           + 52 * (32 + 744 * e2 + 834 * e4 + 119 * e6) * cos6i
                           + 221 * (16 + 156 * e2 + 147 * e4 + 19 * e6) * cos8i) * csc_i**4 * sin3w
            - 32032 * e4 * (22 * (-12 + 43 * e2 + 13 * e4)
                            + (-276 + 2213 * e2 + 605 * e4) * cos2i
                            + (200 + 2462 * e2 + 578 * e4) * cos4i
                            + 17 * (20 + 91 * e2 + 19 * e4) * cos6i) * csc_i**2 * sin5w
            - 4576 * e6 * (-91 + 165 * e2 + 4 * (-7 + 102 * e2) * cos2i
                           + 17 * (7 + 19 * e2) * cos4i) * sin7w
        ) * RJ**9

    if l == 10:  # B.9
        return (1.0 / (2147483648 * a**10 * em1**10)) * 495 * pi * (
            7 * (23814 * (128 + 3 * e2 * (640 + 1568 * e2 + 840 * e4 + 77 * e6))
                 + 294 * (20608 + 307584 * e2 + 751968 * e4 + 402360 * e6 + 36855 * e8) * cos2i
                 + 312 * (19072 + 279936 * e2 + 679392 * e4 + 362040 * e6 + 33075 * e8) * cos4i
                 + 1053 * (5504 + 21 * e2 * (3712 + 8864 * e2 + 4680 * e4 + 425 * e6)) * cos6i
                 + 442 * (12928 + 21 * e2 * (8064 + 18528 * e2 + 9560 * e4 + 855 * e6)) * cos8i
                 + 4199 * (1664 + 3 * e2 * (5760 + 7 * e2 * (1632 + 760 * e2 + 63 * e4))) * cos10i)
            - 84 * (294 * (-192 + 544 * e2 + 6020 * e4 + 4802 * e6 + 539 * e8)
                    + 98 * (-1088 + 5344 * e2 + 42028 * e4 + 32158 * e6 + 3549 * e8) * cos2i
                    + 104 * (-832 + 10720 * e2 + 55356 * e4 + 39102 * e6 + 4165 * e8) * cos4i
                    + 13 * (-3648 + 156640 * e2 + 626444 * e4 + 411278 * e6 + 42245 * e8) * cos6i
                    + 442 * (64 + 7328 * e2 + 24276 * e4 + 14826 * e6 + 1463 * e8) * cos8i
                    + 4199 * (64 + 1312 * e2 + 3220 * e4 + 1666 * e6 + 147 * e8) * cos10i) * cos2w
            - 8736 * e2 * (33 * (-224 + 1208 * e2 + 1600 * e4 + 221 * e6)
                           + 24 * (-448 + 3896 * e2 + 4680 * e4 + 627 * e6) * cos2i
                           + 4 * (-224 + 30040 * e2 + 30432 * e4 + 3825 * e6) * cos4i
                           + 136 * (64 + 920 * e2 + 808 * e4 + 95 * e6) * cos6i
                           + 323 * (32 + 248 * e2 + 192 * e4 + 21 * e6) * cos8i) * cos4w * sin_i**2
            - 1248 * e4 * (26 * (-924 + 3418 * e2 + 855 * e4)
                           + (-23100 + 199186 * e2 + 45747 * e4) * cos2i
                           + 238 * (84 + 850 * e2 + 171 * e4) * cos4i
                           + 323 * (84 + 346 * e2 + 63 * e4) * cos6i) * cos6w * sin_i**4
            - 7072 * e6 * (-120 + 221 * e2 + 4 * (-8 + 133 * e2) * cos2i
                           + 19 * (8 + 21 * e2) * cos4i) * cos8w * sin_i**6
        ) * RJ**10

    raise ValueError(f"zonal order l={l} not implemented (need 2..10)")


def domega_dmu(a, e):
    """d<Delta omega_g>/dmu from the leading GR term: Delta omega_GR = 6 pi mu / (a(1-e^2)c^2).

    a in km, c in km/s; result has units (km^3/s^2)^-1.
    """
    return 6.0 * np.pi / (a * (1.0 - e**2) * C_KM_S**2)


def gradient_vector(a, e, inc, omega, RJ, n_max=10, apply_normalization=True):
    """Gradient d<Delta omega_g>/d(parameter), ordered as [mu, J2, ..., J_{n_max}]
    to match the covariance-matrix slice.

    apply_normalization: if True, multiply each zonal derivative by sqrt(2l+1) to
    convert d/dJ_l (unnormalised) -> d/dJbar_l (fully normalised), so it contracts
    consistently with the fully-normalised covariance. This is the physically
    rigorous choice.
    """
    grad = [domega_dmu(a, e)]
    for l in range(2, n_max + 1):
        d = domega_dJl(l, a, e, inc, omega, RJ)
        if apply_normalization:
            d *= norm_factor(l)
        grad.append(d)
    return np.array(grad)
