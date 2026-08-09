#!/usr/bin/env python3
"""Which findings can actually explain themselves, and which just show a number.

Every `Finding` the detector layer emits is a box the user can click. If there
is no `Explainer` behind that box, the click lands on the measurement restated —
which is the complaint this whole knowledge layer exists to answer. So the gap
has to be visible, and it has to be visible without anyone remembering to look.

The finding ids are read out of `analysis/detectors.py` by parsing it, not from
a list maintained here. A hand-kept list is wrong the first time somebody adds a
detector and does not think to update it, and it would be wrong in the direction
that hides the problem. Three id shapes exist in that file and all three are
resolved:

    id="clipping.hard_clipping"                     a literal
    id="low_end.sub_energy_hot" if hot else "..."   a ternary, both branches
    id=f"frequency_balance.{band.name}_{direction}" a template, expanded

The template case is the one that matters: nine macro bands times two directions
is eighteen findings from a single line of source, and they are the most common
findings in the product.

    python tools/check_knowledge.py            # report, exit 1 on a real gap
    python tools/check_knowledge.py --strict   # any gap at all fails
    python tools/check_knowledge.py --json     # for CI

Exit codes: 0 clean, 1 a critical-capable finding has no explainer (or a fix
step names a capability that does not exist), 2 the source could not be read.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import knowledge  # noqa: E402
from analysis.capabilities import CAPABILITIES  # noqa: E402
from analysis.core import MACRO_BAND_ORDER  # noqa: E402

DETECTORS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "detectors.py"
)

# Placeholders inside an f-string id that cannot be read off the source, keyed by
# the expression exactly as it is written. `band.name` is an attribute on a
# runtime object; the set of values it can take lives in `core.MACRO_BANDS`, and
# taking it from there means adding a macro band automatically widens this check.
_VOCAB: Dict[str, Tuple[str, ...]] = {
    "band.name": tuple(MACRO_BAND_ORDER),
}


class Unresolved(Exception):
    """An id expression this script cannot expand.

    Raised rather than skipped. A finding id we failed to read is a finding we
    would silently report as covered, which is the one outcome worse than a
    missing explainer.
    """


# ---------------------------------------------------------------------------
# Reading the ids out of the detector source
# ---------------------------------------------------------------------------


def _string_bindings(scope: ast.AST) -> Dict[str, Set[str]]:
    """Every string a local name is bound to anywhere in one function.

    Deliberately flow-insensitive: the union over all bindings is exactly what
    we want, because a name bound in a loop over `("drums", "vocals")` and then
    reassigned from an object attribute still only ever reaches the `Finding`
    call holding one of those two values.
    """
    found: Dict[str, Set[str]] = {}

    def add(name: str, value: str) -> None:
        found.setdefault(name, set()).add(value)

    for node in ast.walk(scope):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                    if isinstance(node.value.value, str):
                        add(target.id, node.value.value)
        elif isinstance(node, ast.For):
            _bind_loop(node.target, node.iter, add)

    return found


def _bind_loop(target: ast.AST, source: ast.AST, add) -> None:
    """Bind `for x in (...)` and `for x, y in ((...), (...))`."""
    if not isinstance(source, (ast.Tuple, ast.List)):
        return

    if isinstance(target, ast.Name):
        for element in source.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                add(target.id, element.value)
        return

    if isinstance(target, ast.Tuple):
        for element in source.elts:
            if not isinstance(element, (ast.Tuple, ast.List)):
                continue
            for slot, value in zip(target.elts, element.elts):
                if (
                    isinstance(slot, ast.Name)
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ):
                    add(slot.id, value.value)


def _resolve(node: ast.AST, bindings: Dict[str, Set[str]]) -> Set[str]:
    """Every string one id expression can evaluate to."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}

    if isinstance(node, ast.IfExp):
        return _resolve(node.body, bindings) | _resolve(node.orelse, bindings)

    if isinstance(node, ast.Name):
        values = bindings.get(node.id)
        if not values:
            raise Unresolved(f"name {node.id!r} is not bound to any literal string")
        return set(values)

    if isinstance(node, ast.JoinedStr):
        # Expand the template: each part contributes a set, and the id set is the
        # product of them. Two placeholders of 9 and 2 values is 18 ids.
        parts: List[Sequence[str]] = []
        for piece in node.values:
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                parts.append([piece.value])
            elif isinstance(piece, ast.FormattedValue):
                parts.append(sorted(_resolve_placeholder(piece.value, bindings)))
            else:
                raise Unresolved("unexpected f-string part")
        return {"".join(combo) for combo in itertools.product(*parts)}

    raise Unresolved(f"unsupported id expression: {ast.dump(node)[:120]}")


def _resolve_placeholder(node: ast.AST, bindings: Dict[str, Set[str]]) -> Set[str]:
    source = ast.unparse(node)
    if source in _VOCAB:
        return set(_VOCAB[source])
    try:
        return _resolve(node, bindings)
    except Unresolved:
        raise Unresolved(
            f"placeholder {{{source}}} has no known value set — add it to _VOCAB "
            f"in {os.path.basename(__file__)}"
        ) from None


def _enclosing_scopes(tree: ast.AST) -> Dict[ast.AST, ast.AST]:
    """Map each `Finding(...)` call to the function it is written inside."""
    owner: Dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                if _is_finding_call(inner):
                    owner[inner] = node
    return owner


def _is_finding_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Finding"
    )


def _kwarg(call: ast.Call, name: str) -> Optional[ast.AST]:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _critical_capable(call: ast.Call) -> bool:
    """Whether this finding can ever come back as `critical`.

    A literal severity is exactly what it says. Anything computed — in practice
    `_severity(ratio)`, which returns "critical" above a fixed miss ratio — can
    reach critical, so it counts.
    """
    severity = _kwarg(call, "severity")
    if severity is None:
        return True
    if isinstance(severity, ast.Constant) and isinstance(severity.value, str):
        return severity.value == "critical"
    return True


def emitted_findings(path: str = DETECTORS) -> Tuple[Dict[str, bool], List[str]]:
    """Every finding id in the detector source -> whether it can be critical.

    Also returns the id expressions that could not be resolved, so a new shape
    of id fails loudly instead of quietly shrinking the checked set.
    """
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)

    scopes = _enclosing_scopes(tree)
    cache: Dict[int, Dict[str, Set[str]]] = {}
    findings: Dict[str, bool] = {}
    problems: List[str] = []

    for call, scope in scopes.items():
        id_node = _kwarg(call, "id")
        if id_node is None:
            problems.append(f"line {call.lineno}: Finding(...) with no id=")
            continue

        key = id(scope)
        if key not in cache:
            cache[key] = _string_bindings(scope)

        try:
            ids = _resolve(id_node, cache[key])
        except Unresolved as exc:
            problems.append(f"line {call.lineno}: {exc}")
            continue

        critical = _critical_capable(call)
        for finding_id in ids:
            findings[finding_id] = findings.get(finding_id, False) or critical

    return findings, problems


# ---------------------------------------------------------------------------
# Checking the fix steps themselves
# ---------------------------------------------------------------------------


def bad_capability_slugs(explainers: Dict[str, "knowledge.Explainer"]) -> List[str]:
    """Fix steps whose `needs` is not a real capability.

    A step requiring a slug that does not exist in `capabilities.py` is dead:
    `applicable_steps` can never match it, so every reader either sees the
    `without` fallback forever or, with no fallback, never sees the step at all.
    Silent, and indistinguishable from the step not having been written.
    """
    bad: List[str] = []
    for finding_id, explainer in sorted(explainers.items()):
        for index, step in enumerate(explainer.how_to_fix):
            if step.needs and step.needs not in CAPABILITIES:
                bad.append(f"{finding_id} step {index + 1}: needs={step.needs!r}")
    return bad


def steps_without_fallback(explainers: Dict[str, "knowledge.Explainer"]) -> List[str]:
    """Gated steps that vanish entirely for a producer without the tool.

    Not an error — some moves genuinely need the box — but worth counting, since
    a finding whose every step is gated shows an empty fix list to someone on
    stock plugins.
    """
    hidden: List[str] = []
    for finding_id, explainer in sorted(explainers.items()):
        steps = explainer.how_to_fix
        if steps and all(s.needs and not s.without for s in steps):
            hidden.append(finding_id)
    return hidden


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report() -> Dict[str, object]:
    explainers = knowledge.ensure_registered()
    findings, problems = emitted_findings()

    direct: List[str] = []
    inherited: List[str] = []
    missing: List[str] = []
    for finding_id in sorted(findings):
        if finding_id in explainers:
            direct.append(finding_id)
        elif knowledge.explain(finding_id) is not None:
            # Covered only by the dimension fallback in `explain()`. Real
            # coverage, but generic teaching where a specific answer was wanted.
            inherited.append(finding_id)
        else:
            missing.append(finding_id)

    critical_missing = [f for f in missing if findings[f]]
    orphans = sorted(set(explainers) - set(findings))

    return {
        "emitted": len(findings),
        "explainers": len(explainers),
        "direct": direct,
        "inherited": inherited,
        "missing": missing,
        "critical_missing": critical_missing,
        "orphans": orphans,
        "unresolved": problems,
        "bad_capability_slugs": bad_capability_slugs(explainers),
        "fully_gated": steps_without_fallback(explainers),
    }


def _print(report: Dict[str, object]) -> None:
    emitted = report["emitted"]
    direct = report["direct"]
    covered = len(direct) + len(report["inherited"])

    print(f"detectors emit {emitted} finding ids; {report['explainers']} explainers registered")
    print(f"  direct explainer     {len(direct):>3}")
    print(f"  dimension fallback   {len(report['inherited']):>3}")
    print(f"  no explainer         {len(report['missing']):>3}")
    pct = (100.0 * covered / emitted) if emitted else 100.0
    print(f"  coverage             {pct:.1f}%")

    for finding_id in report["inherited"]:
        print(f"    fallback only: {finding_id}")

    for finding_id in report["missing"]:
        mark = "CRITICAL-capable" if finding_id in report["critical_missing"] else "non-critical"
        print(f"    MISSING ({mark}): {finding_id}")

    for line in report["unresolved"]:
        print(f"    UNRESOLVED id in detectors.py: {line}")

    for line in report["bad_capability_slugs"]:
        print(f"    BAD needs= slug: {line}")

    if report["orphans"]:
        print(f"  {len(report['orphans'])} explainer(s) no detector emits:")
        for finding_id in report["orphans"]:
            print(f"    orphan: {finding_id}")

    if report["fully_gated"]:
        print(f"  {len(report['fully_gated'])} finding(s) whose every fix step needs a tool:")
        for finding_id in report["fully_gated"]:
            print(f"    all steps gated: {finding_id}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail on non-critical gaps and on fallback-only coverage",
    )
    args = parser.parse_args(argv)

    try:
        report = build_report()
    except OSError as exc:
        print(f"could not read the detector source: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print(report)

    failed = bool(
        report["critical_missing"] or report["unresolved"] or report["bad_capability_slugs"]
    )
    if args.strict:
        failed = failed or bool(report["missing"] or report["inherited"])

    if failed and not args.json:
        print("\nFAIL: a finding a user can see has nothing to teach them.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
