"""Fetch every resource link and fail if any of them is not real.

This is what makes the no-fabrication rule in `knowledge.Resource` enforceable
rather than aspirational. A link is the one claim on the page a user can check
in a second, so a dead one costs more trust than the resource was worth.

    python tools/check_links.py            # check everything
    python tools/check_links.py --offline  # structure only, no network

Three checks, in order of how badly they fail:

1. **Host allowlist.** A `reference` pointing anywhere outside
   `ALLOWED_RESOURCE_HOSTS` is a hard failure — that is the fabrication guard.
2. **YouTube is search-only.** A `/watch?v=...` link means somebody pasted a
   specific video id, which is exactly what rots. Also a hard failure.
3. **It resolves.** Every `reference` is fetched. `search` URLs are checked
   structurally, since a search page cannot 404 by construction.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import knowledge as K  # noqa: E402

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"


def _fetch(url: str) -> Tuple[int, str]:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, ""
    except urllib.error.HTTPError as exc:
        # 403 is usually a bot block rather than a dead page (tech.ebu.ch does
        # this). Report it, but do not treat it as fatal — a human can check.
        return exc.code, "blocked?" if exc.code in (401, 403, 429) else ""
    except Exception as exc:  # noqa: BLE001
        return 0, type(exc).__name__


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="skip network checks")
    args = ap.parse_args()

    K.ensure_registered() if hasattr(K, "ensure_registered") else None

    hard: List[str] = []
    warn: List[str] = []
    to_fetch: List[Tuple[str, str]] = []
    counts = {"search": 0, "reference": 0}
    with_resources = 0

    for fid, exp in sorted(K.EXPLAINERS.items()):
        resources = getattr(exp, "resources", ()) or ()
        if resources:
            with_resources += 1
        for r in resources:
            counts[r.kind] = counts.get(r.kind, 0) + 1
            host = (urlparse(r.url).hostname or "").lower()

            if host not in K.ALLOWED_RESOURCE_HOSTS:
                hard.append(f"{fid}: host not allowlisted -> {host or '(none)'} · {r.url}")
                continue
            if host == "www.youtube.com" and "/results" not in r.url:
                hard.append(f"{fid}: YouTube link is not a search -> {r.url}")
                continue
            if not r.label.strip():
                hard.append(f"{fid}: resource has no label -> {r.url}")
            # Attribution is enforced, not trusted. A reference without a named
            # publisher is a link the reader cannot judge the provenance of.
            if r.kind == "reference" and not r.source.strip():
                hard.append(f"{fid}: reference has no source/attribution -> {r.url}")
            if r.kind == "reference":
                to_fetch.append((fid, r.url))

    if to_fetch and not args.offline:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda p: (p[0], p[1], *_fetch(p[1])), to_fetch))
        for fid, url, code, note in results:
            if code == 200:
                continue
            line = f"{fid}: HTTP {code or 'ERR'} {note} -> {url}"
            (warn if note == "blocked?" else hard).append(line)

    total = len(K.EXPLAINERS)
    print(f"  explainers                {total}")
    print(f"  with resources            {with_resources}  ({with_resources / max(total, 1) * 100:.0f}%)")
    print(f"  search links              {counts.get('search', 0)}")
    print(f"  reference links           {counts.get('reference', 0)}  "
          f"({'not fetched' if args.offline else f'{len(to_fetch)} fetched'})")

    for line in warn:
        print(f"  WARN  {line}")
    for line in hard:
        print(f"  FAIL  {line}")

    if hard:
        print(f"\n{len(hard)} broken or disallowed link(s). Nothing ships with a dead link.")
        return 1
    print("\nall resource links ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
