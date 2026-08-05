"""Operator tooling that sits beside the product but is not part of the request path.

Nothing in here is imported by `main.py`, the API routers or `analysis.*`. These
are command-line utilities an engineer runs by hand against a local corpus, so
they are free to be slow, to use a process pool, and to write files — none of
which would be acceptable inside a request.

    python tools/calibrate.py --genre trap --input ~/Music/trap-references/

Run them from `backend/` so `import analysis...` resolves; each script also
inserts the backend directory on `sys.path` so `python tools/calibrate.py` and
`python -m tools.calibrate` both work.
"""

from __future__ import annotations

__all__: list = []
