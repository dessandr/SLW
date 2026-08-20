# Stage-input templates

These are syntax templates, not material defaults. Replace every mesh, atom
index, orbital slice, path, and file name with values from the calculation at
hand. One input file runs one `calculation`; use separate exchange inputs for
static `calculation='j'` and dynamic `calculation='dj'` products. Set
`ltensor=.true.` in `&exchange` when the full tensor form is required.

Validate a template without opening any scientific input file:

```bash
slw_exchange.x -in examples/exchange.in --dry-run
```
