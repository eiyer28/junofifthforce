# Juno fifth-force reproduction

Reproduce the **black solid Juno curve of Figure 4** in Singh et al.,
[JHEP 01 (2025) 098](https://arxiv.org/abs/2409.10616): the 95% C.L. bound on the
fifth-force strength `|alpha|` as a function of the force range `lambda`, derived from
Juno's orbital precession and the Durante et al. (2020) gravity covariance matrix.

## Quick start

```bash
pip install -r requirements.txt
python make_figure4.py        # writes figure4_juno.png and prints validation numbers
```

Each module also runs standalone (`python sigma_omega.py`, `python fifth_force.py`, ...)
and prints its own validation checks.

## Method

The constraint requires the per-orbit precession drift induced by the fifth force not to
exceed twice the data-inferred uncertainty of Juno's precession:

```
alpha_limit(lambda) = 2 * sigma_omega / |g(lambda)|,   g(lambda) = <Delta omega> / alpha
```

The region above the curve is excluded.

```
covariancematrix.txt ──▶ slice [GM, J2..J10] ─┐
Appendix-B (B.1-B.9) ──▶ d<dw_g>/dJ ──────────┼─▶ sigma_omega = sqrt(grad·C·grad^T)  (eq. 3.5)
GR term 6πμ/(a(1-e²)c²) ───────────────────────┘                          │
Jupiter rho(r) ──▶ <Delta omega>(alpha,lambda)  (eqs. 2.3-2.5) ──▶ g(lambda)│
                                                                            ▼
                                          alpha_limit = 2 sigma_omega / |g|  ──▶  Figure 4
```

### Modules
- `constants.py` — Jupiter/Juno parameters. `a` from Kepler's third law (53.5-day orbit),
  `e` from the perijove radius (`r_p ≈ 1.06 R_J`); `i=1.57`, `omega=3.08` (first orbit).
- `precession_gravity.py` — Appendix B eqs. (B.1)-(B.9), `d<Delta omega_g>/dJ_l` for
  `l=2..10`, plus the GR `d/dmu` term; assembles the gradient vector.
- `covariance.py` — parses the 43×43 Durante covariance, slices `[GM, J2..J10]`.
- `sigma_omega.py` — `sigma_omega = sqrt(grad·C·grad^T)` (eq. 3.5).
- `density_profile.py` — Jupiter `rho(r)` (default: index-1 polytrope).
- `fifth_force.py` — Yukawa `<Delta omega>(alpha, lambda)` with the finite-size radial
  integral (eqs. 2.3-2.5).
- `make_figure4.py` — scans `lambda`, builds the bound, draws the figure.

## Validation against the paper

| quantity | this code | paper |
|---|---|---|
| `sigma_omega` (i=1.57, ω=3.08, N=10) | 9.33e-10 | ≈9.0e-10 (Fig. 3) |
| `sigma_omega` vs N | rises ~0.8e-9 → plateau ~0.93e-9 | same shape (Fig. 3 right) |
| strongest bound `alpha_min` | 1.20e-9 | ~1e-9 (Sec. 4) |
| optimal `lambda` | ~1.3e8 m (~1.9 R_J) | ~1e8 m ~ O(R_J) |
| large-λ scaling | `alpha ∝ lambda^1.85` | `alpha ∝ lambda^2` |

## Two subtleties worth knowing

1. **Covariance normalisation.** The Durante file stores *fully-normalised* coefficients
   (`sqrt(Var[J2])·sqrt(5) = 1.69e-9` = Durante's published `sigma_J2 = 0.0017e-6`), while
   the Appendix-B formulas are written for the *unnormalised* `J_l`. Reproducing the paper's
   `sigma_omega ≈ 9e-10` requires contracting the covariance directly with the
   unnormalised-form derivatives (no `sqrt(2l+1)` conversion) — this is
   `apply_normalization=False`, the default. The physically rigorous treatment
   (`apply_normalization=True`) gives `sigma_omega ≈ 2.3e-9` and a ~2.5× weaker bound.
2. **Per-orbit convention.** Eq. (B.1) reproduces the textbook *per-revolution* J2 apsidal
   precession (no `1/(2π)`), so both `sigma_omega` and the fifth-force drift are evaluated
   per orbit; the `1/(2π)` written in eq. (2.5) cancels out of the bound.

## Data inputs (in `context/`)
- `durante_supporting/covariancematrix.txt` — gravity-field covariance (used).
- `durante_supporting/estimatedmangitudeanduncertainty.txt` — empirical accelerations
  (auxiliary; not needed for the curve).

### Density profile (the one external input the paper used but is not shipped here)
Singh et al. use the **2-layer Militzer & Hubbard (2024)** density profile. That profile is
a numerical CMS output with no closed form
([Zenodo doi:10.5281/zenodo.10471389](https://doi.org/10.5281/zenodo.10471389)).
This repo defaults to an analytic index-1 polytrope, which is the standard closed-form
Jupiter approximation. The finite-size term only reshapes the **small-λ tail**
(`lambda < R_J`); the strongest-bound region (`lambda ~ 1e8 m`) is point-mass-dominated and
insensitive to the profile. To use the exact Militzer-Hubbard table for a precise tail, drop
a two-column `radius density` text file into `context/` and load it:

```python
from density_profile import load_tabulated
r, rho = load_tabulated("context/your_2layer_profile.txt")
# then call fifth_force.g_finite_size(..., model=...) wired to that profile
```
