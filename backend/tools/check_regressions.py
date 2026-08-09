#!/usr/bin/env python
"""The scoring contract, as an executable claim.

Three properties, and they are the ones that break silently when somebody
retunes the severity or penalty constants:

1. **Own-genre baseline.** A finished reference record measured against its own
   genre scores in the low 90s. If this drifts down, the scorer has started
   treating normal records as broken.

2. **Cross-genre gradient.** The *same* trap master measured against
   trap/pop/rock/folk/ambient falls monotonically, and falls a long way. This is
   the one that flattened: a defect/deviation split that softens a deviation
   with a flat multiplier, on top of a penalty table that has already bucketed
   every deviation into one severity, throws the genre signal away instead of
   softening it — and the failure looks like three genres scoring identically.
   The spread assertion is what catches that; monotonicity alone does not.

3. **A genuinely broken file still scores badly.** `mix_problem.wav` clips. No
   amount of "maybe they meant it" may rescue it.

Run: backend/venv/bin/python -m tools.check_regressions   (from backend/)
Exit status is the test result, so this drops into CI unchanged.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.engine import analyze_mix  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testdata")

# Tolerance on each figure. Wide enough that a real DSP improvement does not
# trip it, narrow enough that a collapsed gradient cannot hide inside it.
TOL = 4.0

OWN_GENRE = [
    ("reference_trap.wav", "trap", 90.1),
    ("reference_pop.wav", "pop", 90.2),
    ("reference_folk.wav", "folk", 91.8),
]

GRADIENT = [("trap", 90.1), ("pop", 87.6), ("rock", 76.4), ("folk", 71.8), ("ambient", 65.1)]

BROKEN = [("mix_problem.wav", "trap", 47.2), ("mix_problem.wav", "pop", 41.8)]

# A gradient that does not span at least this much is not a gradient. The
# observed failure produced 90.7 / 90.1 / 82.9 / 82.9 / 82.9 — monotonic by a
# hair, and useless.
MIN_GRADIENT_SPAN = 20.0

# No two adjacent genres in the gradient may be indistinguishable.
MIN_ADJACENT_STEP = 1.0


def main() -> int:
    failures = []

    print("own-genre baseline")
    for name, genre, expect in OWN_GENRE:
        got = analyze_mix(os.path.join(FIXTURES, name), genre).health_score
        bad = abs(got - expect) > TOL
        failures.append(f"{name}@{genre}: {got} vs {expect}") if bad else None
        print(f"  {name:22} @{genre:8} {got:5.1f}  (expect {expect:5.1f}) {'FAIL' if bad else 'ok'}")

    print("\ncross-genre gradient, reference_trap.wav")
    scores = []
    for genre, expect in GRADIENT:
        got = analyze_mix(os.path.join(FIXTURES, "reference_trap.wav"), genre).health_score
        scores.append(got)
        bad = abs(got - expect) > TOL
        failures.append(f"gradient@{genre}: {got} vs {expect}") if bad else None
        print(f"  {genre:8} {got:5.1f}  (expect {expect:5.1f}) {'FAIL' if bad else 'ok'}")

    span = scores[0] - scores[-1]
    if span < MIN_GRADIENT_SPAN:
        failures.append(f"gradient span {span:.1f} < {MIN_GRADIENT_SPAN} — genre signal lost")
    print(f"  span {span:5.1f}  (need >= {MIN_GRADIENT_SPAN}) "
          f"{'FAIL' if span < MIN_GRADIENT_SPAN else 'ok'}")

    for (g0, _), (g1, _), a, b in zip(GRADIENT, GRADIENT[1:], scores, scores[1:]):
        if b > a + 0.05:
            failures.append(f"gradient not monotonic: {g0} {a} -> {g1} {b}")
        elif a - b < MIN_ADJACENT_STEP:
            failures.append(f"gradient flat between {g0} and {g1}: {a} -> {b}")

    print("\nbroken file still scores badly")
    for name, genre, expect in BROKEN:
        got = analyze_mix(os.path.join(FIXTURES, name), genre).health_score
        bad = abs(got - expect) > TOL
        failures.append(f"{name}@{genre}: {got} vs {expect}") if bad else None
        print(f"  {name:22} @{genre:8} {got:5.1f}  (expect {expect:5.1f}) {'FAIL' if bad else 'ok'}")

    print()
    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("all scoring regressions pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
