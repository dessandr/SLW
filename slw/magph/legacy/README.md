# Legacy magnon--phonon implementation

This directory quarantines the magnon--phonon code retained from the initial
SLW extraction. It is kept for numerical parity and for post-processing tools
while the public workflow is exposed through `slw_magph.x` and `slw_post.x`.

Layout:

- `reference/` contains the ten retained calculation drivers selected by the
  public stage registry.
- The remaining modules are shared numerical helpers, compatibility adapters,
  diagnostics, and plotting tools used by those drivers.
- `plotting/` contains the retained `plot.in` compatibility implementation.

Old imports such as `slw.magph.hybrid` and direct invocations such as
`python -m slw.magph.hybrid` are intentionally not preserved. These modules
are implementation details and may move again as native calculation engines
replace them.
