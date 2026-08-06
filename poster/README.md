# Symposium poster

LaTeX source for a **40 in (W) × 30 in (H)** landscape poster on the Juno
fifth-force project.

## Files
- `poster.tex` — the poster (beamerposter, custom size 101.6 cm × 76.2 cm).
- `figures/` — figures embedded in the poster (copied from the project root).

## Build

Requires a TeX distribution (TeX Live / MiKTeX) with `beamer` and `beamerposter`.

```bash
pdflatex poster.tex
pdflatex poster.tex   # run twice so captions/refs settle
```

This produces `poster.pdf` at the exact board size (40 × 30 in). Print at 100%
(no "fit to page" scaling).

## Branding
- Colors follow Brown University's Visual Identity: Red `#ED1C24`,
  Seal Brown `#4E3629`, Gold `#FFC72C` (defined as `brownred`, `brownseal`,
  `browngold` in `poster.tex`).
- Authors: Eashan Iyer, Ash Lassonde, Praniti Singh, JiJi Fan (all with a
  superscript `1` = Department of Physics, Brown University).
- The header uses a **single** copy of the **official** two-color stacked Brown
  Department of Physics logo. To avoid transparent PNG regions rendering as
  black, the logo is flattened onto white
  (`figures/brown_physics_logo_white.png`, generated from the original with PIL:
  `alpha_composite` over a white canvas). Size is controlled by `\brownwordmark`
  near the top of `poster.tex` (`width=0.82\linewidth`).

## Editing
- `scale=1.35` in the `beamerposter` options controls the global font size;
  raise it for larger text, lower it if content overflows.
- The typographic wordmark uses Palatino (`ppl`); swap the official asset in for
  print-quality output.
