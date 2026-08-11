# Report

Working LaTeX draft of the paper. Zip **this folder** and upload to Overleaf; set the main document to `main.tex` and compile with pdfLaTeX (run twice for the TOC and bibliography).

| file | role |
|---|---|
| `main.tex` | narrative (single file) |
| `preamble.tex` | packages and macros |
| `refs.bib` | bibliography |
| `figures/` | plots copied from the repo root |

Regenerate figures with the root scripts (`make_figure4.py`, `compare_constraints.py`, …), then copy the PNGs into `figures/` so Overleaf stays self-contained. Physics, usage, and a results summary live in the [project README](../README.md).
