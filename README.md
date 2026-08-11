# Juno fifth-force constraints

NASA's Juno spacecraft has orbited Jupiter since 2016 on a highly elliptical path, from just above the cloud tops to large distances. That geometry makes the orbit a sensitive probe of any long-range **fifth force** between Juno and Jupiter — a Yukawa-like correction to gravity that appears in many extensions of the Standard Model.

Such a force would add a small extra **precession** of Juno's argument of perijove. Gravity-field uncertainty already allows some precession, so requiring the extra drift to stay within that uncertainty limits the force's strength \(\alpha\) as a function of its range \(\lambda\).

This repository **reproduces** the Juno bound of Singh et al. ([JHEP 01 (2025) 098](https://arxiv.org/abs/2409.10616)), which used the Durante et al. (2020) gravity solution, and **updates** it with later Juno gravity data (Kaspi et al. 2023 and a re-derived high-degree covariance). The physics and the constraint formula are unchanged; what improved is the measured Jovian field.

Derivations, covariance conventions, and a full discussion of limitations belong in the paper (`report/`). This README is a high-level map of the result and how to rerun it.

## Results so far

All numbers below are 95% C.L. at Juno's first-orbit geometry (perijove 1), unless noted.

**Reproduction.** Contracting the Durante (2020) covariance with the precession response recovers the published Juno curve:

| quantity | this analysis | Singh et al. |
|---|---|---|
| precession uncertainty \(\sigma_\omega\) | \(9.33\times 10^{-10}\) | \(\approx 9.0\times 10^{-10}\) |
| strongest bound \(\alpha_{\min}\) | \(1.20\times 10^{-9}\) | \(\sim 10^{-9}\) |
| best-constrained range \(\lambda\) | \(\sim 1.3\times 10^8\,\mathrm{m}\) (\(\sim 1.9\,R_J\)) | \(\sim 10^8\,\mathrm{m}\) |

**Update.** Newer gravity solutions tighten the same bound. How much depends on how the old and new covariances are compared:

| comparison | \(\sigma_\omega\) (old \(\to\) new) | improvement |
|---|---|---|
| same-slice overlay (Durante vs Kaspi PJ37) | \(9.33\times 10^{-10}\to 3.52\times 10^{-10}\) | **2.65×** |
| basis-consistent (Durante vs `cov_220824`) | \(2.32\times 10^{-9}\to 1.74\times 10^{-9}\) | **1.34×** |

The 2.65× figure is what you get by feeding both published matrices through the paper's original contraction. The 1.34× figure matches each matrix to its verified harmonic convention before comparing. Both are useful; they answer different questions. The paper discusses which number to quote as a “final” improvement.

**What actually sets the bound**

- **Low-degree gravity harmonics dominate**, especially \(J_2\). High-degree zonals barely move \(\sigma_\omega\) because Juno's semi-major axis is \(\sim 57\,R_J\), so the response falls as \((R_J/a)^l\).
- With the modern covariance, \(\sigma_\omega\) **plateaus by \(N\sim 4\)** (J2 alone is already at the percent level). The older Durante matrix keeps rising until \(N\sim 12\) because its high-degree uncertainties are larger and its neighbour-to-neighbour correlations add constructively.
- **Orbit geometry matters modestly.** Across Juno's realised \((\omega, i)\), \(\sigma_\omega\) spans roughly \(7.1\times 10^{-10}\)–\(9.0\times 10^{-10}\); PJ01 sits near the high end and is the conservative default.
- The **correlation structure** of the gravity solution has changed: modern matrices have weaker low-degree off-diagonals, so extra harmonics no longer pile onto the uncertainty the way they did in Durante.

The peak sensitivity remains near \(\lambda\sim R_J\). Improving Jupiter's gravity field therefore lowers the whole exclusion curve without shifting where it is strongest.

## How to run the analysis

The intended interface is the notebook **`juno_pipeline_comparison.ipynb`**. It is self-contained: constants, the precession response, a polytropic Jupiter density, and the covariance blocks are all embedded. The only Python dependencies are NumPy and Matplotlib.

### Setup

```bash
pip install -r requirements.txt
```

Then open the notebook in Jupyter, VS Code / Cursor, or Google Colab (it was developed with Colab metadata) and **run all cells from top to bottom**. Later cells reuse functions and arrays defined earlier; skipping the first few will break the rest.

A full run takes on the order of a minute on a laptop. Some later diagnostic cells do not have cached outputs in the file — re-running those cells regenerates the plots.

### What the notebook does, in order

Work through it as a single pipeline. You can stop early if you only need the reproduction.

1. **Orbit and constants** — Juno's semi-major axis and eccentricity from the 53.5-day period and perijove radius.
2. **Gravity response** — how a change in each zonal harmonic \(J_l\) (plus the GR/\(GM\) term) shifts the argument of perijove.
3. **Durante covariance \(\to \sigma_\omega\)** — slice the published gravity covariance and contract it with that response. This is the uncertainty that sets the bound.
4. **Fifth-force drift** — Yukawa precession vs coupling \(\alpha\) and range \(\lambda\) (notebook Figure 2).
5. **Reproduced exclusion curve** — 95% C.L. \(\alpha(\lambda)\) from Durante, matching Singh et al. Figure 4.
6. **Kaspi overlay** — same pipeline on the PJ37 gravity solution; prints the 2.65× same-slice improvement.
7. **Truncation in \(N\)** — \(\sigma_\omega\) vs how many harmonics are kept (Singh et al. Figure 3, right), for both matrices.
8. **Normalization check and `cov_220824`** — verify how each covariance is stored, then recompute a basis-consistent comparison (the 1.34× factor).
9. **Structure diagnostics** — why the modern matrix plateaus so fast (J2 dominance, diagonal vs off-diagonal budget, correlation patterns).
10. **Orbit geometry** — \(\sigma_\omega\) in the \((\omega, i)\) plane, per-perijove values, and an improvement-factor map.

If a plot looks empty, the cell almost certainly was not executed in this session — run it (and its predecessors).

### Optional: regenerate paper/poster figures from scripts

The notebook is enough to reproduce every result above. The `make_*.py` / `compare_constraints.py` scripts write PNG files used by `report/` and `poster/`:

```bash
python make_figure4.py              # reproduced exclusion curve
python compare_constraints.py       # Durante vs Kaspi overlay
python make_figure2.py              # fifth-force drift
python make_figure3_right.py        # sigma_omega vs N
python make_sigma_omega_checks.py   # geometry and improvement-factor maps
```

Copy updated PNGs into `report/figures/` and `poster/figures/` if you are rebuilding those documents.

## Paper and poster

- **`report/`** — LaTeX write-up (Overleaf: zip the folder, set `main.tex` as the main document). This is where method, conventions, and caveats are spelled out.
- **`poster/`** — symposium poster source.

## References

- P. Singh et al., *JHEP* 01 (2025) 098, [arXiv:2409.10616](https://arxiv.org/abs/2409.10616)
- D. Durante et al., *Geophys. Res. Lett.* **47** (2020)
- Y. Kaspi et al., *Nat. Astron.* **7**, 1463 (2023)
