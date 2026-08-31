# LaTeX Theory Manuscript

`main.tex` is a paper-style derivation of the package formalism. It is intentionally explicit about what is source-derived and what is a project extension.

Build:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Output in this archive:

- `wannier_torque_formalism.pdf`
- modular section files under `sections/`
- `references.bib` (manuscript citation subset; the package-wide master database is `../references/references.bib`)
- `SOURCE_MAP.md`

The author macro in `main.tex` is a placeholder and should be edited before submission.
