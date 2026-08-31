# Theory Manuscript Map

The paper-style derivation is under `theory/`.

| Manuscript section | Implementation documents |
|---|---|
| Spinor Hamiltonian and exchange field | docs/03 |
| Local projection and torque vertices | docs/04 |
| Magnetic-force-theorem mixed response | docs/05 |
| Finite-q gauge and q-pair completion | docs/06 |
| Phonon projection | docs/07 |
| External magnon projection | docs/08 |
| Validation | docs/12 |
| Source and novelty boundary | docs/00, docs/18, references/ |

Build with:

```bash
cd theory
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The generated `wannier_torque_formalism.pdf` is included in the package. `SOURCE_MAP.md` maps manuscript equations and claims to exact literature sources or labels them as project extensions.
