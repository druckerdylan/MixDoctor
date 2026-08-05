"""Pure DSP: audio in, measured numbers out.

Every module here returns one of the `*Measurement` models from `..types` and
knows nothing about genres, thresholds or verdicts — `targets.py` is off-limits
below this line. That separation is what lets the detector layer be argued with
without anybody questioning the measurements underneath it.

Public functions are resolved lazily (PEP 562). Importing `analysis.dsp` never
imports a module you didn't ask for, so a slow or missing sibling can't break
an unrelated import, and new modules are picked up without editing this file.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, List

# Known entry points, so `from analysis.dsp import measure_stereo` resolves
# without a directory scan. Anything not listed still resolves via the scan.
_ENTRY_POINTS: Dict[str, str] = {
    "measure_clipping": "clipping",
    "measure_loudness": "loudness",
    "measure_spectral": "spectral",
    "measure_stereo": "stereo",
    "measure_phase": "phase",
    "measure_dynamics": "dynamics",
    "measure_transients": "transients",
    "measure_low_end": "lowend",
    "measure_vocal": "vocal",
    "measure_clarity": "clarity",
    "measure_sections": "sections",
    # Heavyweight and opt-in: naming it here keeps the resolver from walking
    # (and importing) every sibling to find it, but the import itself still
    # only happens when somebody actually asks for stems.
    "measure_stems": "separation",
}

__all__ = sorted(_ENTRY_POINTS)


def _submodules() -> List[str]:
    return sorted(m.name for m in pkgutil.iter_modules(__path__))


def __getattr__(name: str):
    """Resolve a public measurement function from its submodule on first use."""
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name = _ENTRY_POINTS.get(name)
    candidates = [module_name] if module_name else _submodules()

    for candidate in candidates:
        try:
            module = importlib.import_module(f".{candidate}", __name__)
        except ImportError:
            continue
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value  # cache: __getattr__ runs once per name
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(__all__) | set(globals()) | set(_submodules()))
