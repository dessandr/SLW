# Legacy exchange quarantine

This directory contains every exchange module that predates the native typed
engine: numerical helpers, diagnostics, post-processing tools, SOC builders,
and the old calculation drivers. Nothing here is part of the public
`slw.exchange` API.

Production exchange calculations enter `slw.exchange.engine`. The six drivers
temporarily used for numerical parity are isolated under `reference/` and are
called only through `reference/adapter.py`; their argparse `main()` paths are
not used by `slw_exchange.x`. Retained `slw_post.x`, EPC, and SOC workflows may
still import specific legacy helpers until native replacements exist.

Historical package-root module paths and compatibility shims are intentionally
not preserved. Once a kernel has a synthetic regression test and a native
implementation, its legacy copy can be removed without changing the public
input format.
