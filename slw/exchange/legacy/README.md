# Legacy exchange quarantine

This directory contains every exchange module that predates the native typed
engine: numerical helpers, diagnostics, post-processing tools, SOC builders,
and the old calculation drivers. Nothing here is part of the public
`slw.exchange` API.

Production exchange calculations enter `slw.exchange.engine` and dispatch only
to `slw.exchange.kernels`. Wrappers under `reference/` delegate from archived
tools toward native kernels; the dependency never points from the public
engine back into this directory. Retained `slw_post.x` diagnostics may still
enter specific archived modules until native replacements exist.

Historical package-root module paths and compatibility shims are intentionally
not preserved. The public scalar/tensor J and dJ kernels have generated
regression fixtures and serial/MPI parity tests, so archive removal will not
change the public input format.
