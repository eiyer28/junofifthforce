"""Jupiter radial mass-density profile rho(r) for the finite-size fifth-force integral.

Singh et al. use the 2-layer density profile of Militzer & Hubbard (2024). That
profile is a numerical output of the concentric-Maclaurin-spheroid (CMS) method
(tabulated data on Zenodo, doi:10.5281/zenodo.10471389); it has no closed form and
is not shipped in this repository.

This module therefore provides:

  * an analytic default profile -- the index-1 polytrope -- which is the standard
    closed-form approximation to Jupiter's interior (Jupiter is well described by an
    n=1 polytrope), normalised to Jupiter's mass and reference radius; and
  * a loader for a tabulated rho(r) if you drop a data file into `context/`.

The finite-size correction only reshapes the SMALL-lambda tail (lambda < R_J);
the strongest-bound region (lambda ~ 1e8 m) is in the point-mass regime and is
insensitive to the detailed profile. See README for how to plug in the exact
Militzer-Hubbard table for a precise tail.
"""

import numpy as np

from constants import M_JUP_KG, R_J_M


def _normalise_to_mass(r, rho, total_mass=M_JUP_KG):
    """Rescale rho so that the spherical mass integral equals total_mass."""
    m = np.trapezoid(4.0 * np.pi * r**2 * rho, r)
    return rho * (total_mass / m)


def polytrope_n1(n_points=4000, R=R_J_M, total_mass=M_JUP_KG):
    """Index-1 polytrope: rho(r) = rho_c * sinc(pi r / R), zero at the surface.

    Returns (r [m], rho [kg/m^3]) with the mass normalised to total_mass.
    """
    r = np.linspace(0.0, R, n_points)
    x = np.pi * r / R
    # sin(x)/x with the x->0 limit handled.
    shape = np.ones_like(x)
    nz = x > 0
    shape[nz] = np.sin(x[nz]) / x[nz]
    shape = np.clip(shape, 0.0, None)
    rho = _normalise_to_mass(r, shape, total_mass)
    return r, rho


def uniform(n_points=4000, R=R_J_M, total_mass=M_JUP_KG):
    """Constant-density sphere (sanity-check profile)."""
    r = np.linspace(0.0, R, n_points)
    rho = np.full_like(r, total_mass / (4.0 / 3.0 * np.pi * R**3))
    return r, rho


def two_layer(core_frac=0.2, core_mass_frac=0.1, n_points=4000, R=R_J_M,
              total_mass=M_JUP_KG):
    """Crude 2-layer model: constant-density core (radius core_frac*R holding
    core_mass_frac of the mass) plus an n=1-polytrope-shaped envelope.

    Provided for experimentation with the Militzer-Hubbard 2-layer concept; the
    polytrope default is the recommended analytic profile.
    """
    r = np.linspace(0.0, R, n_points)
    Rc = core_frac * R
    rho = np.zeros_like(r)

    core = r <= Rc
    rho_core = core_mass_frac * total_mass / (4.0 / 3.0 * np.pi * Rc**3)
    rho[core] = rho_core

    env = ~core
    x = np.pi * r[env] / R
    rho[env] = np.sin(x) / x
    rho[env] = _normalise_to_mass(r[env], rho[env],
                                  (1.0 - core_mass_frac) * total_mass) \
        if env.any() else rho[env]
    # Re-normalise the whole thing to be safe against discretisation.
    rho = _normalise_to_mass(r, rho, total_mass)
    return r, rho


def load_tabulated(path, n_points=4000, R=R_J_M, total_mass=M_JUP_KG):
    """Load a tabulated rho(r) from a 2-column text file (radius, density).

    Radii are rescaled so the outer edge maps to R, densities are renormalised to
    total_mass. Use this to plug in the exact Militzer-Hubbard 2-layer profile.
    """
    data = np.loadtxt(path)
    r_raw, rho_raw = data[:, 0], data[:, 1]
    order = np.argsort(r_raw)
    r_raw, rho_raw = r_raw[order], rho_raw[order]
    r = np.linspace(0.0, R, n_points)
    rho = np.interp(r, r_raw / r_raw.max() * R, rho_raw)
    rho = _normalise_to_mass(r, rho, total_mass)
    return r, rho


def get_profile(model="polytrope", **kwargs):
    """Dispatch to a density model by name."""
    models = {
        "polytrope": polytrope_n1,
        "uniform": uniform,
        "two_layer": two_layer,
    }
    if model not in models:
        raise ValueError(f"unknown density model {model!r}; choose from {list(models)}")
    return models[model](**kwargs)


if __name__ == "__main__":
    for name in ("polytrope", "uniform", "two_layer"):
        r, rho = get_profile(name)
        m = np.trapezoid(4.0 * np.pi * r**2 * rho, r)
        print(f"{name:10s}: central rho = {rho[0]:8.1f} kg/m^3, "
              f"mass check = {m:.4e} kg (target {M_JUP_KG:.4e})")
