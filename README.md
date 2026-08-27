# Updating fifth-force constraints from Juno

NASA’s Juno spacecraft orbits Jupiter on a highly elliptical path. A hypothetical fifth force between Juno and Jupiter would make that orbit precess anomalously relative to the standard model and known physical effects. We can put bounds on the precession for the closest point of approach using space probe data. How tightly we can bound that force depends on the precesion with which we can determine Jupiter's gravitational field. 

This repository reproduces an earlier Juno bound ([Singh et al., JHEP 01 (2025) 098](https://arxiv.org/abs/2409.10616)) and then updates it with an improved gravitational field using additional atmospheric modeling techniques ([Kaspi et al., 2023](https://www.nature.com/articles/s41550-023-02077-8)).

- Physics and results: [`Juno_Mission_UTRA_Writeup.pdf`](Juno_Mission_UTRA_Writeup.pdf)
- Code/analysis: [`juno_pipeline_comparison.ipynb`](juno_pipeline_comparison.ipynb)

## Results

The Kaspi (2023) field reduces the precession uncertainty by about **1.34×**, and tightens the fifth-force bound by about **25%** near its strongest point. Details in write-up **§6**.

| | Durante 2020 (PJ01–17) | Kaspi 2023 (through PJ37) |
|---|---|---|
| Precession uncertainty σ<sub>ω</sub> | 2.32×10<sup>−9</sup> | 1.74×10<sup>−9</sup> |
| Strongest 95% bound on \|α\| | ~3.0×10<sup>−9</sup> | ~2.2×10<sup>−9</sup> |

α is the fifth-force strength relative to gravity; λ is its range. The bound is strongest near **λ ≈ 1.35×10<sup>8</sup> m** (about two Jupiter radii). Values of \|α\| above the curve are excluded.

## Run the notebook

```bash
pip install -r requirements.txt
```

Then open `juno_pipeline_comparison.ipynb` and run all cells. The only Python dependencies are NumPy and Matplotlib. Constants and covariance data are already in the notebook and the data folders below.

## Read the write-up by topic

[`Juno_Mission_UTRA_Writeup.pdf`](Juno_Mission_UTRA_Writeup.pdf) is the place for derivations, caveats, and figures.

| If you want… | See |
|---|---|
| Why spacecraft orbits can test extra forces | **§1** |
| What α and λ mean, and the Yukawa potential | **§2.1** |
| Why Jupiter’s finite size and gravity harmonics matter | **§2.2–2.3** |
| The two gravity datasets (Durante vs Kaspi) | **§3** |
| Gravity covariance → precession error → bound on α(λ) | **§4** |
| Coefficient normalization (why the numbers above differ from a naïve copy of the earlier paper) | **§4.2** |
| Reproduction check, high-degree harmonics, correlations, orbital geometry | **§5** |
| The updated bound | **§6** |
| Limitations | **§7** |

## What’s in the repo

| Path | Role |
|---|---|
| `juno_pipeline_comparison.ipynb` | Reproduction, comparison, and diagnostics |
| `Juno_Mission_UTRA_Writeup.pdf` | Write-up |
| `context/durante_supporting/` | Durante et al. (2020) gravity covariance |
| `solution_ref_GRAVtoPJ37+PJ01_normal-modes_jnCnstr-0.1mGal/` | Kaspi et al. (2023) PJ37 gravity solution |
| `J_calc.nb` | Mathematica source for the high-degree zonal precession derivatives |
