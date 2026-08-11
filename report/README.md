# Report (Overleaf-ready)

Zip **this folder** (`report/`) and upload to Overleaf. Set the main document to `main.tex`, then press **Recompile** (pdfLaTeX).

## Layout

```
report/
  main.tex                 ← Overleaf main file
  preamble.tex
  refs.bib
  sections/
    abstract.tex
    introduction.tex
    theory.tex
    data.tex
    method.tex
    results.tex
    discussion.tex
    conclusions.tex
    acknowledgements.tex
  figures/
  README.md
```

## Writing

Fill gray `\writeme{...}` prompts in `sections/*.tex` (start with `results.tex`). Keep `sections/abstract.tex` unless you intentionally revise it.

Regenerate figures from the repo root scripts, then copy into `figures/` so Overleaf stays self-contained.
