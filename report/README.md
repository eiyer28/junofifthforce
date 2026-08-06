# Project report (LaTeX)

Working draft of a written report on the Juno fifth-force project.

## Contents

| File | Role |
|---|---|
| `report.tex` | Main narrative draft (LaTeX) |
| `figures/` | Figures referenced by the draft |

## Build

```bash
cd report
pdflatex report.tex
pdflatex report.tex   # second pass for TOC / refs
```

Requires a TeX distribution with `amsmath`, `booktabs`, `graphicx`, and `hyperref`.

## How to extend

Edit `report.tex` as results mature. Figures are copies from the project root / poster; regenerate them with the usual scripts (`make_figure4.py`, `compare_constraints.py`, `make_sigma_omega_checks.py`, …) and re-copy into `figures/` when numbers change.

Suggested next sections: formal abstract polish, density-profile systematics, recommended “final” bound under one stated normalization convention, and acknowledgements.
