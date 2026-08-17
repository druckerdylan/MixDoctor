"""The downloadable document: everything the analysis knows, written to be read.

The web report is an interface — you click a dimension, a panel expands, a fix
appears. A document is not. It has to work in one pass, on a phone, on a printed
page, six weeks later, for somebody who was not sitting in front of the app when
it was generated. That changes what it has to contain and the order it has to
arrive in.

Two readers, one file
---------------------
Someone learning and someone who already knows both asked for this, and the
temptation is to write for the middle, which serves neither. The answer here is
structure, not compromise. **Teaching sits at the front**: the three moves that
matter, then the defects in plain language with numbered steps, then the
reference deviations framed as decisions. **Reference material sits at the
back**: the concepts behind everything that fired, then a table with every
measurement, its unit and its genre target. A beginner reads until it stops
being useful; an engineer starts at the appendix. Neither is condescended to and
nothing is watered down.

Defects and deviations, again
-----------------------------
The document inherits the split from `analysis.detectors.finding_kind`, and it
is more load-bearing here than anywhere else in the product, because a document
is read as a verdict in a way a UI panel is not. Section 3 is *what is wrong*:
clipping, inter-sample overs, an inverted channel, a mix that vanishes in mono.
Section 4 is *where this differs from the reference*, and every entry in it
carries what the departure costs **and what it buys**, plus the case in which it
is the right call. A producer who tucked the topline on purpose should finish
section 4 feeling understood, not corrected.

Format
------
Markdown is the source of truth and HTML is rendered from the same block tree,
so the two cannot drift. The HTML is a single self-contained file — inline CSS,
no scripts required, no fonts, no images, no network of any kind — with print
rules that make "Save as PDF" from a browser produce a clean document. There is
deliberately no PDF library here: it would be a large dependency to reproduce
something every browser already does well.

No AI is required. Where `analysis.engineer` ran, its prescriptions are woven
into sections 3 and 4 and marked as its work; where it did not, the deterministic
explainers carry the whole document on their own. That is the entire reason the
explainer layer exists.
"""

from __future__ import annotations

import html as _html
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from analysis import capabilities, knowledge, targets
from analysis.detectors import finding_kind
from analysis.types import (
    DIMENSION_LABELS,
    SEVERITY_RANK,
    TRACK_INTENT_LABELS,
    Evidence,
    Finding,
    Measurements,
    MixAnalysis,
    OwnedPlugin,
    Prescription,
)

__all__ = ["render_markdown", "render_html", "suggested_filename"]


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def _fin(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


def _num(value: object, nd: int = 2) -> str:
    """A number a human reads, not a float repr.

    Trailing zeros go, thousands get separators, and anything non-finite
    renders as an em dash rather than `nan` — a table cell reading `nan` looks
    like a bug in the analyser even when the measurement was simply not taken.
    """
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if v != v or v in (float("inf"), float("-inf")):
        return "—"
    if abs(v) >= 1000 and nd <= 1:
        return f"{v:,.0f}"
    text = f"{v:.{nd}f}"
    # Only trim inside the fraction. `"{:.0f}".format(99.5)` is "100", and a
    # blanket rstrip("0") turns that into "1" — which is how a 99.5 score
    # rendered as a 1 in the first draft of the appendix.
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text not in ("", "-", "-0") else "0"


def _signed(value: object, nd: int = 1) -> str:
    """Signed, but only when the *rendered* value is non-zero.

    `+0` is noise: it claims a direction the rounding just threw away.
    """
    v = _fin(value)
    text = _num(v, nd)
    return f"+{text}" if v > 0 and text != "0" else text


def _lead_sentence(text: str) -> Tuple[str, str]:
    """Split "Do this. Because that." into (instruction, rest).

    Fix steps in the knowledge layer are sometimes a whole paragraph in the
    `action` slot — especially the `without` fallbacks, which have to carry
    their own reasoning. Bolding the paragraph makes a numbered list unreadable,
    so the imperative becomes the step and the reasoning drops to the detail
    line. The lookbehind keeps decimals ("-1.0 dBTP", "Q 0.7") intact.
    """
    body = " ".join(str(text or "").split())
    parts = re.split(r'(?<=[a-z)\]"”])\.\s+(?=[A-Z])', body, maxsplit=1)
    if len(parts) == 2 and len(parts[0]) >= 12:
        return parts[0] + ".", parts[1]
    return body, ""


def _pct(value: object, nd: int = 0) -> str:
    return f"{_num(_fin(value) * 100.0, nd)}%"


def _window(w: Optional[Sequence[float]], unit: str = "", nd: int = 1) -> str:
    if not w or len(w) < 2:
        return "—"
    suffix = f" {unit}" if unit else ""
    return f"{_num(w[0], nd)} to {_num(w[1], nd)}{suffix}"


def _yes(flag: object) -> str:
    return "yes" if bool(flag) else "no"


def _minutes(total: int) -> str:
    if total <= 0:
        return "—"
    if total < 60:
        return f"{total} min"
    hours, rest = divmod(total, 60)
    return f"{hours} h" if rest == 0 else f"{hours} h {rest} min"


def _sentence(text: str) -> str:
    """Trim, collapse whitespace and guarantee terminal punctuation."""
    out = " ".join(str(text or "").split())
    if out and out[-1] not in ".!?:;":
        out += "."
    return out


def _clip(text: str, limit: int) -> str:
    """Bound a client-supplied string without cutting mid-word."""
    out = " ".join(str(text or "").split())
    if len(out) <= limit:
        return out
    return out[:limit].rsplit(" ", 1)[0] + "…"


def _slug(text: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return out or "section"


# ---------------------------------------------------------------------------
# Document model
#
# One block tree, two renderers. Building an intermediate representation rather
# than emitting Markdown and then converting it means the HTML can carry things
# Markdown has no syntax for — page breaks, cards, a two-column cover — while
# both outputs stay guaranteed to contain the same content.
# ---------------------------------------------------------------------------


@dataclass
class Block:
    pass


@dataclass
class Heading(Block):
    level: int
    text: str
    #: Small uppercase line above the heading. Section number, kind, dimension.
    eyebrow: str = ""


@dataclass
class Para(Block):
    text: str
    #: Renders larger in HTML. For the sentence that opens a section.
    lead: bool = False


@dataclass
class Bullets(Block):
    items: List[str]


@dataclass
class Steps(Block):
    """A numbered procedure. Each item is (action, why-it-works)."""

    items: List[Tuple[str, str]]


@dataclass
class Checklist(Block):
    """(task, time estimate). Ticks are printable boxes, not interactive."""

    items: List[Tuple[str, str]]


@dataclass
class Table(Block):
    headers: List[str]
    rows: List[List[str]]
    #: Column indices to right-align. Numbers read wrong left-aligned.
    numeric: Tuple[int, ...] = ()


@dataclass
class Callout(Block):
    """A boxed aside. `tone` drives colour in HTML and a label in Markdown."""

    tone: str  # defect | reference | good | note | ai | verdict | question | choice
    title: str
    body: List[str] = field(default_factory=list)


@dataclass
class Definitions(Block):
    """Key/value pairs. The cover block, and the per-finding fact strip."""

    pairs: List[Tuple[str, str]]
    compact: bool = False


@dataclass
class Rule(Block):
    pass


@dataclass
class PageBreak(Block):
    """HTML/print only. Markdown has no page, so it renders as nothing."""


class Doc:
    """A growing block list, plus the one bit of state the prose needs.

    `term()` is that state. The brief asks that a term be defined the first time
    it appears and never again, which is a property of the whole document rather
    than of any one sentence — so the document owns it. Call sites write
    ``doc.term("LUFS", "the loudness measurement streaming platforms use")`` and
    get the definition once.
    """

    def __init__(self) -> None:
        self.blocks: List[Block] = []
        self._defined: set = set()

    # -- building ---------------------------------------------------------
    def add(self, block: Block) -> "Doc":
        self.blocks.append(block)
        return self

    def h(self, level: int, text: str, eyebrow: str = "") -> "Doc":
        return self.add(Heading(level, text, eyebrow))

    def p(self, text: str, lead: bool = False) -> "Doc":
        text = " ".join(str(text or "").split())
        if text:
            self.add(Para(text, lead))
        return self

    def ul(self, items: Iterable[str]) -> "Doc":
        kept = [" ".join(str(i).split()) for i in items if str(i).strip()]
        if kept:
            self.add(Bullets(kept))
        return self

    def steps(self, items: Sequence[Tuple[str, str]]) -> "Doc":
        kept = [(a, d) for a, d in items if str(a).strip()]
        if kept:
            self.add(Steps(kept))
        return self

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[str]],
              numeric: Tuple[int, ...] = ()) -> "Doc":
        body = [[str(c) for c in row] for row in rows]
        if body:
            self.add(Table(list(headers), body, numeric))
        return self

    def note(self, tone: str, title: str, body: Sequence[str] = ()) -> "Doc":
        self.add(Callout(tone, title, [" ".join(str(b).split()) for b in body if str(b).strip()]))
        return self

    # -- vocabulary -------------------------------------------------------
    def term(self, word: str, definition: str) -> str:
        """The word, with its definition in parentheses the first time only."""
        key = word.strip().lower()
        if key in self._defined:
            return word
        self._defined.add(key)
        return f"{word} ({definition})"

    def seen(self, word: str) -> None:
        """Mark a term as already defined — for terms an explainer defines."""
        self._defined.add(word.strip().lower())


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def _md_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _to_markdown(doc: Doc) -> str:
    out: List[str] = []

    for block in doc.blocks:
        if isinstance(block, Heading):
            if block.eyebrow:
                out.append(f"*{block.eyebrow}*")
                out.append("")
            out.append(f"{'#' * max(1, min(6, block.level))} {block.text}")
            out.append("")

        elif isinstance(block, Para):
            out.append(block.text)
            out.append("")

        elif isinstance(block, Bullets):
            out.extend(f"- {item}" for item in block.items)
            out.append("")

        elif isinstance(block, Steps):
            for i, (action, detail) in enumerate(block.items, 1):
                out.append(f"{i}. **{action}**")
                if detail:
                    out.append(f"   {detail}")
            out.append("")

        elif isinstance(block, Checklist):
            for task, when in block.items:
                suffix = f" — *{when}*" if when else ""
                out.append(f"- [ ] {task}{suffix}")
            out.append("")

        elif isinstance(block, Table):
            out.append("| " + " | ".join(_md_cell(h) for h in block.headers) + " |")
            out.append(
                "|"
                + "|".join(
                    (" ---: " if i in block.numeric else " --- ")
                    for i in range(len(block.headers))
                )
                + "|"
            )
            for row in block.rows:
                cells = list(row) + [""] * (len(block.headers) - len(row))
                out.append("| " + " | ".join(_md_cell(c) for c in cells) + " |")
            out.append("")

        elif isinstance(block, Callout):
            label = _CALLOUT_LABEL.get(block.tone, "Note")
            out.append(f"> **{label} — {block.title}**")
            for line in block.body:
                out.append(">")
                out.append(f"> {line}")
            out.append("")

        elif isinstance(block, Definitions):
            for key, value in block.pairs:
                out.append(f"- **{key}:** {value}")
            out.append("")

        elif isinstance(block, Rule):
            out.append("---")
            out.append("")

        elif isinstance(block, PageBreak):
            continue

    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


_CALLOUT_LABEL: Dict[str, str] = {
    "defect": "Defect",
    "reference": "Differs from reference",
    "good": "Working",
    "note": "Note",
    "ai": "From the engineer write-up",
    # The score headline, an unanswered question, and a decision the producer
    # has already made. Each needed its own label because the existing five all
    # assert something: "Working — 1 defect to fix before mastering" and
    # "Working — is the bottom octave meant to be this big?" are both wrong in
    # the same way, which is that the label contradicts the sentence.
    "verdict": "Verdict",
    "question": "Open question",
    "choice": "Your decision",
}


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")


def _inline(text: str) -> str:
    """Escape, then re-apply the three inline marks the document uses.

    Escaping happens first and unconditionally, so nothing carried in from the
    client payload can open a tag. The marks are applied to already-escaped
    text, which is why this is safe despite looking like string templating.
    """
    out = _html.escape(str(text), quote=False)
    out = _INLINE_CODE.sub(r"<code>\1</code>", out)
    out = _BOLD.sub(r"<strong>\1</strong>", out)
    out = _ITALIC.sub(r"<em>\1</em>", out)
    return out


_CSS = """
:root{
  --ink:#14161a; --ink-2:#3d434d; --ink-3:#6a7280; --line:#e2e5ea;
  --bg:#ffffff; --panel:#f7f8fa; --accent:#0f6d5a;
  --defect:#a3231f; --defect-bg:#fdf3f2; --defect-line:#f0cfcd;
  --ref:#8a5a12; --ref-bg:#fdf8ef; --ref-line:#efe0c4;
  --good:#0f6d5a; --good-bg:#f0f8f5; --good-line:#c9e4da;
  --ai:#2f4d86; --ai-bg:#f3f6fc; --ai-line:#d3ddef;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
}
.wrap{max-width:52rem;margin:0 auto;padding:3rem 1.5rem 5rem}
h1,h2,h3,h4{line-height:1.2;letter-spacing:-.015em;margin:0}
h1{font-size:2.4rem;font-weight:700}
h2{font-size:1.6rem;font-weight:650;margin:3.2rem 0 .9rem;padding-top:1.4rem;border-top:2px solid var(--ink)}
h3{font-size:1.16rem;font-weight:650;margin:2.1rem 0 .5rem}
h4{font-size:.98rem;font-weight:650;margin:1.4rem 0 .35rem;color:var(--ink-2)}
p{margin:0 0 1rem}
p.lead{font-size:1.08rem;color:var(--ink-2)}
.eyebrow{
  font:600 11px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  letter-spacing:.13em;text-transform:uppercase;color:var(--ink-3);margin:0 0 .35rem
}
h2+.eyebrow,h3+.eyebrow{margin-top:-.3rem}
ul,ol{margin:0 0 1rem;padding-left:1.35rem}
li{margin:0 0 .42rem}
code{
  font:.88em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:.05em .35em
}
hr{border:0;border-top:1px solid var(--line);margin:2.2rem 0}
a{color:var(--accent)}

/* cover */
.cover{border:2px solid var(--ink);border-radius:10px;padding:2rem 2rem 1.6rem;margin-bottom:2.4rem}
.cover h1{margin:.5rem 0 .2rem}
.cover .sub{font-size:1.05rem;color:var(--ink-2);margin:.1rem 0 1.4rem}
.scorebar{display:flex;flex-wrap:wrap;align-items:baseline;gap:.55rem;margin:0 0 1.3rem}
.score{font:700 3.4rem/1 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;letter-spacing:-.04em}
.of{color:var(--ink-3);font-size:1rem}
.grade{
  margin-left:.4rem;border:1.5px solid var(--ink);border-radius:6px;
  padding:.15rem .55rem;font-weight:700;font-size:1.05rem
}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(11rem,1fr));gap:.75rem 1.6rem;margin:0}
.facts div{min-width:0}
.facts dt{
  font:600 10.5px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3)
}
.facts dd{margin:.12rem 0 0;font-size:.95rem;overflow-wrap:anywhere}

/* callouts */
.callout{border:1px solid var(--line);border-left-width:4px;border-radius:6px;
  background:var(--panel);padding:.85rem 1.1rem;margin:0 0 1.15rem}
.callout .t{font-weight:650;margin:0 0 .3rem}
.callout p{margin:0 0 .5rem;font-size:.95rem;color:var(--ink-2)}
.callout p:last-child{margin-bottom:0}
.callout .k{
  font:600 10.5px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  letter-spacing:.12em;text-transform:uppercase;display:block;margin-bottom:.2rem
}
.c-defect{background:var(--defect-bg);border-color:var(--defect-line);border-left-color:var(--defect)}
.c-defect .k{color:var(--defect)}
.c-reference{background:var(--ref-bg);border-color:var(--ref-line);border-left-color:var(--ref)}
.c-reference .k{color:var(--ref)}
.c-good{background:var(--good-bg);border-color:var(--good-line);border-left-color:var(--good)}
.c-good .k{color:var(--good)}
.c-ai{background:var(--ai-bg);border-color:var(--ai-line);border-left-color:var(--ai)}
.c-ai .k{color:var(--ai)}
.c-note{border-left-color:var(--ink-3)}
.c-note .k{color:var(--ink-3)}
.c-verdict{background:var(--panel);border-color:var(--line);border-left-color:var(--ink)}
.c-verdict .k{color:var(--ink)}
.c-verdict .t{font-size:1.1rem}
.c-question{background:var(--ref-bg);border-color:var(--ref-line);border-left-color:var(--ref)}
.c-question .k{color:var(--ref)}
.c-choice{background:var(--good-bg);border-color:var(--good-line);border-left-color:var(--good)}
.c-choice .k{color:var(--good)}

/* steps + checklist */
ol.steps{list-style:none;counter-reset:s;padding-left:0;margin:0 0 1.2rem}
ol.steps>li{counter-increment:s;position:relative;padding-left:2.3rem;margin:0 0 .85rem}
ol.steps>li::before{
  content:counter(s);position:absolute;left:0;top:.02rem;width:1.6rem;height:1.6rem;
  border:1.5px solid var(--ink);border-radius:50%;display:flex;align-items:center;
  justify-content:center;font:600 .78rem/1 ui-monospace,Menlo,monospace
}
ol.steps .why{display:block;color:var(--ink-2);font-size:.93rem;margin-top:.18rem}
ul.check{list-style:none;padding-left:0;margin:0 0 1.2rem}
ul.check>li{display:flex;gap:.7rem;align-items:flex-start;padding:.45rem 0;border-bottom:1px solid var(--line)}
ul.check .box{flex:0 0 auto;width:1rem;height:1rem;border:1.5px solid var(--ink-3);border-radius:3px;margin-top:.28rem}
ul.check .task{flex:1 1 auto;min-width:0}
ul.check .when{
  flex:0 0 auto;color:var(--ink-3);
  font:600 11px/1.7 ui-monospace,Menlo,monospace;letter-spacing:.06em;white-space:nowrap
}

/* tables */
.tw{overflow-x:auto;margin:0 0 1.3rem;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{text-align:left;padding:.44rem .7rem;border-bottom:1px solid var(--line);vertical-align:top}
th{
  font:600 10.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);
  border-bottom:1.5px solid var(--ink)
}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tbody tr:nth-child(even){background:var(--panel)}

/* definition strips */
dl.strip{display:flex;flex-wrap:wrap;gap:.35rem 1.5rem;margin:0 0 1rem;
  padding:.6rem .9rem;background:var(--panel);border:1px solid var(--line);border-radius:6px}
dl.strip div{display:flex;gap:.4rem;align-items:baseline;font-size:.9rem}
dl.strip dt{color:var(--ink-3);margin:0}
dl.strip dd{margin:0;font-weight:600}

.toolbar{max-width:52rem;margin:1.2rem auto -1.6rem;padding:0 1.5rem;text-align:right}
.toolbar button{
  font:600 12px/1 ui-monospace,Menlo,monospace;letter-spacing:.08em;text-transform:uppercase;
  padding:.6rem 1rem;border:1.5px solid var(--ink);border-radius:6px;background:#fff;
  color:var(--ink);cursor:pointer
}
.foot{margin-top:3rem;padding-top:1.2rem;border-top:1px solid var(--line);
  color:var(--ink-3);font-size:.82rem}

@media (max-width:640px){
  .wrap{padding:2rem 1.1rem 3rem}
  h1{font-size:1.9rem}.score{font-size:2.8rem}.cover{padding:1.4rem}
}

@media print{
  @page{margin:16mm 14mm}
  html,body{background:#fff}
  .wrap{max-width:none;padding:0}
  .toolbar,.no-print{display:none !important}
  body{font-size:10.5pt;line-height:1.5}
  h1{font-size:22pt}h2{font-size:15pt}h3{font-size:12pt}
  h2,h3,h4{break-after:avoid;page-break-after:avoid}
  .pb{break-before:page;page-break-before:always}
  .callout,tr,ol.steps>li,ul.check>li,dl.strip{break-inside:avoid;page-break-inside:avoid}
  .tw{overflow:visible}
  tbody tr:nth-child(even){background:#f4f5f7 !important;-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .callout{-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .cover{break-after:page;page-break-after:always}
}
"""


def _to_html(doc: Doc, title: str) -> str:
    body: List[str] = []

    for block in doc.blocks:
        if isinstance(block, Heading):
            lvl = max(1, min(6, block.level))
            if block.eyebrow:
                body.append(f'<p class="eyebrow">{_inline(block.eyebrow)}</p>')
            body.append(f"<h{lvl}>{_inline(block.text)}</h{lvl}>")

        elif isinstance(block, Para):
            cls = ' class="lead"' if block.lead else ""
            body.append(f"<p{cls}>{_inline(block.text)}</p>")

        elif isinstance(block, Bullets):
            items = "".join(f"<li>{_inline(i)}</li>" for i in block.items)
            body.append(f"<ul>{items}</ul>")

        elif isinstance(block, Steps):
            items = []
            for action, detail in block.items:
                why = f'<span class="why">{_inline(detail)}</span>' if detail else ""
                items.append(f"<li><strong>{_inline(action)}</strong>{why}</li>")
            body.append('<ol class="steps">' + "".join(items) + "</ol>")

        elif isinstance(block, Checklist):
            items = []
            for task, when in block.items:
                w = f'<span class="when">{_inline(when)}</span>' if when else ""
                items.append(
                    '<li><span class="box" aria-hidden="true"></span>'
                    f'<span class="task">{_inline(task)}</span>{w}</li>'
                )
            body.append(f'<ul class="check">{"".join(items)}</ul>')

        elif isinstance(block, Table):
            def _cls(i: int) -> str:
                return ' class="n"' if i in block.numeric else ""

            head = "".join(
                "<th" + _cls(i) + ">" + _inline(h) + "</th>"
                for i, h in enumerate(block.headers)
            )
            rows = []
            for row in block.rows:
                cells = list(row) + [""] * (len(block.headers) - len(row))
                rows.append(
                    "<tr>"
                    + "".join(
                        "<td" + _cls(i) + ">" + _inline(c) + "</td>"
                        for i, c in enumerate(cells)
                    )
                    + "</tr>"
                )
            body.append(
                '<div class="tw"><table><thead><tr>' + head + "</tr></thead><tbody>"
                + "".join(rows) + "</tbody></table></div>"
            )

        elif isinstance(block, Callout):
            label = _CALLOUT_LABEL.get(block.tone, "Note")
            paras = "".join(f"<p>{_inline(line)}</p>" for line in block.body)
            body.append(
                f'<div class="callout c-{_html.escape(block.tone)}">'
                f'<span class="k">{_inline(label)}</span>'
                f'<p class="t">{_inline(block.title)}</p>{paras}</div>'
            )

        elif isinstance(block, Definitions):
            if block.compact:
                items = "".join(
                    f"<div><dt>{_inline(k)}</dt><dd>{_inline(v)}</dd></div>"
                    for k, v in block.pairs
                )
                body.append(f'<dl class="strip">{items}</dl>')
            else:
                items = "".join(
                    f"<div><dt>{_inline(k)}</dt><dd>{_inline(v)}</dd></div>"
                    for k, v in block.pairs
                )
                body.append(f'<dl class="facts">{items}</dl>')

        elif isinstance(block, Rule):
            body.append("<hr>")

        elif isinstance(block, PageBreak):
            body.append('<div class="pb" aria-hidden="true"></div>')

    # A full document, not a fragment: this file is opened on its own, and
    # without a doctype every browser renders it in quirks mode — a different
    # box model, and print margins that do not match what the CSS asks for.
    # `charset` leads the head so the title's bytes are decoded correctly.
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_html.escape(title)}</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        '<div class="toolbar no-print">'
        '<button type="button" onclick="window.print()">Print / Save as PDF</button>'
        "</div>\n"
        f'<main class="wrap">{"".join(body)}</main>\n'
        "</body>\n</html>\n"
    )


# ---------------------------------------------------------------------------
# What a departure from the reference costs, and what it buys
#
# This is the table that stops the document reading as a list of accusations.
# A deviation is a difference from a genre profile, and the honest way to report
# one is to say what you give up, what you get, and the case in which taking it
# is the right call. Keyed by finding id; `_BAND_TRADEOFF` covers the eighteen
# per-band `frequency_balance.*` ids without eighteen near-identical entries.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tradeoff:
    costs: str
    buys: str
    #: A concrete situation in which the departure is correct. Named, not hedged.
    right_when: str = ""


_TRADEOFFS: Dict[str, Tradeoff] = {
    "loudness.too_loud": Tradeoff(
        costs=(
            "Transient life, and you pay for it twice. The peaks were spent buying level, "
            "and then every streaming platform turns the level back down to its own target — "
            "so the listener receives the compromise without receiving the loudness."
        ),
        buys=(
            "Density and immediacy. A hotter master keeps more of itself audible on a phone "
            "speaker or in a noisy car, and inside a genre where every release is pushed this "
            "hard, matching that push is part of sounding current rather than dated."
        ),
        right_when=(
            "You are cutting for a club system, a DJ pool or physical media — anywhere nothing "
            "normalises and the level in the file is the level people hear."
        ),
    ),
    "loudness.cannot_reach_level": Tradeoff(
        costs=(
            "The track arrives quieter than what plays before and after it, which listeners "
            "read as 'smaller' long before they judge anything about the mix itself."
        ),
        buys=(
            "Headroom, and a master a mastering engineer can still do something with. Level is "
            "the one thing that can always be added later; the dynamics spent getting it cannot "
            "be put back."
        ),
        right_when=(
            "This is a mix bound for mastering rather than a finished master. Then the level is "
            "somebody else's job and this line is information, not a task."
        ),
    ),
    "dynamic_range.squashed": Tradeoff(
        costs=(
            "Contrast. When everything is the same size, nothing is big — the chorus cannot "
            "lift because the verse is already at the ceiling, and the ear stops tracking the "
            "arrangement after about thirty seconds."
        ),
        buys=(
            "A wall. Constant density reads as power on small speakers and in a loud room, and "
            "for music built on a loop rather than on an arc, that consistency is the point."
        ),
        right_when=(
            "The record is a single sustained texture by design — a lot of techno, drill and "
            "noise-adjacent production is supposed to arrive flat and stay there."
        ),
    ),
    "dynamic_range.unmastered": Tradeoff(
        costs=(
            "Nothing, technically. It will simply sound quiet and unfinished next to anything "
            "released, and anyone A/B-ing it without matching levels will hear that as a "
            "worse mix rather than an unmastered one."
        ),
        buys=(
            "Every option. An unmastered mix with real headroom is the best possible thing to "
            "hand to a mastering engineer, or to yourself on a different day."
        ),
        right_when="You have not mastered it yet. Which, if this fired, you probably have not.",
    ),
    "dynamic_range.no_section_lift": Tradeoff(
        costs=(
            "The payoff. If the biggest moment measures the same as the verse before it, the "
            "arrangement is doing work the level is refusing to confirm."
        ),
        buys=(
            "Relentlessness. A track that never lifts also never drops, and some music wants "
            "to be a single unbroken pressure rather than a story."
        ),
        right_when=(
            "The energy change is carried by arrangement and texture instead of level — a new "
            "layer entering, the hats doubling — which is a legitimate and very common choice."
        ),
    ),
    "compression.micro_dynamics_lost": Tradeoff(
        costs=(
            "The attack of every hit. Level survives, shape does not, and a drum with no shape "
            "reads as loud rather than as hard."
        ),
        buys=(
            "Glue and sustain. Flattening the peak brings the body and the room up with it, "
            "which is how a mix becomes one object instead of a stack of parts."
        ),
        right_when=(
            "The sound you are after is compressed by definition — heavily crushed drums are "
            "a genre signature, not an accident, in plenty of hip-hop and electronic music."
        ),
    ),
    "compression.pumping": Tradeoff(
        costs=(
            "Steadiness. When the whole mix breathes with the kick, quiet parts get pulled up "
            "and down with it and the noise floor moves audibly."
        ),
        buys=(
            "Groove you can feel. Sidechain pumping is a rhythmic effect in its own right and "
            "in house, EDM and a lot of pop it is deliberately extreme."
        ),
        right_when="The pumping is on the beat and you put it there. Then it is the sound.",
    ),
    "compression.stem_drums_flat": Tradeoff(
        costs="Impact at the source, which no amount of master-bus work can restore.",
        buys="Density and consistency on the drum bus, and a kit that sits still under everything else.",
        right_when="You are after a smashed, parallel-crushed kit sound and printed it that way.",
    ),
    "compression.stem_vocals_flat": Tradeoff(
        costs="Expression. A vocal with no movement stops sounding like a person in a room.",
        buys="Intelligibility at every level, with no word ever disappearing under the track.",
        right_when="The delivery is a flat, deadpan monotone and the compression is matching it.",
    ),
    "mud.low_mid_buildup": Tradeoff(
        costs=(
            "Separation. Energy stacked between the bass and the mids covers everything above "
            "it, so the mix reads as 'covered' or 'boxy' and no single EQ move fixes it."
        ),
        buys=(
            "Warmth and weight. The low mids are where body lives, and a mix cut too hard here "
            "sounds thin and brittle — which is a worse failure and much harder to hear."
        ),
        right_when=(
            "The record is meant to sound close and thick — soul, lo-fi, a lot of 90s hip-hop — "
            "where a scooped low mid would sound clinical and wrong."
        ),
    ),
    "harshness.upper_mid_edge": Tradeoff(
        costs=(
            "Endurance. This is the region the ear is most sensitive to, so energy here buys "
            "attention for thirty seconds and fatigue for the three minutes after that."
        ),
        buys=(
            "Cut. The 2-5 kHz band is what makes a mix audible on a laptop, a phone and a car "
            "stereo, and backing it off too far is how a mix ends up sounding polite."
        ),
        right_when=(
            "The track is built on aggression and is meant to be uncomfortable, or the extra "
            "edge is carrying one element that has to win — a lead, a snare, a hook."
        ),
    ),
    "harshness.sibilance": Tradeoff(
        costs=(
            "Comfort on the consonants. S and T sounds that jump ahead of the words they belong "
            "to are the first thing to become unbearable on earbuds and cheap speakers."
        ),
        buys=(
            "Diction and air. A de-essed vocal loses articulation as well as harshness, and an "
            "over-de-essed one lisps — which is much more obviously wrong than a bright one."
        ),
        right_when=(
            "The voice is meant to be right in your face and the brightness is the character. "
            "Check on earbuds before deciding; this one lies on studio monitors."
        ),
    ),
    "harshness.bright_transients": Tradeoff(
        costs=(
            "Headroom and ear comfort at the top. Bursty high-frequency transients — hats, "
            "shakers, rim clicks — chew through a limiter and spit on lossy encoding."
        ),
        buys=(
            "The crisp, forward top end that a lot of modern production is built on. Hats that "
            "cut through are a deliberate signature, not a mistake."
        ),
        right_when=(
            "The top of the beat is supposed to be sharp and the hats are supposed to sting. "
            "Worth confirming this is the hats and not a vocal before touching anything."
        ),
    ),
    "clarity.congested": Tradeoff(
        costs=(
            "Followability. Every part is present, but no single one can be tracked through the "
            "arrangement, so the mix reads as busy rather than full."
        ),
        buys=(
            "Density. A packed spectrum sounds large and expensive when it works, and the "
            "sparse alternative can sound underproduced."
        ),
        right_when=(
            "The wall is the aesthetic — shoegaze, wall-of-sound production, heavily layered "
            "trap — where individual separation was never the goal."
        ),
    ),
    "clarity.stem_masking": Tradeoff(
        costs="One element losing to another in a specific band, every time they play together.",
        buys="Thickness where the two overlap, which can be exactly the blend you wanted.",
        right_when="The two parts are meant to fuse into one sound rather than stay separate.",
    ),
    "transients.no_punch": Tradeoff(
        costs=(
            "Physical impact. Drums that measure loud but do not stand above their own sustain "
            "stop being felt, and a track people cannot feel is a track they do not move to."
        ),
        buys=(
            "Cohesion and loudness. Peaks are what limits your level, so trading them buys "
            "density — and a very smoothed transient is a real production style, not a fault."
        ),
        right_when=(
            "The kit is meant to sit inside the track rather than in front of it — loops, "
            "sampled breaks and lo-fi drums are frequently soft on purpose."
        ),
    ),
    "low_end.kick_bass_collision": Tradeoff(
        costs=(
            "Definition down low. When the kick's fundamental and the bass note share a "
            "frequency they sum unpredictably — loud on one note, gone on another — and the "
            "bottom of the mix stops being reliable across systems."
        ),
        buys=(
            "One low-end object instead of two. A fused kick-and-808 that moves as a single "
            "weight is the sound of a great deal of modern rap, and separating them too "
            "cleanly can make the low end sound thin and polite."
        ),
        right_when=(
            "The 808 *is* the kick — same sample, tuned to the key — which is a standard "
            "production choice rather than a collision to be untangled."
        ),
    ),
    "low_end.sub_energy_hot": Tradeoff(
        costs=(
            "Headroom, mostly. Sub energy is expensive: it eats the level that the parts people "
            "can actually hear on a phone were going to use."
        ),
        buys=(
            "Weight in a car and on a club system, which for bass-led music is not a bonus — "
            "it is the product."
        ),
        right_when=(
            "The track is made for systems that can reproduce it, and you have confirmed the "
            "sub is musical rather than rumble on something that can play 30 Hz."
        ),
    ),
    "low_end.sub_energy_thin": Tradeoff(
        costs="Scale. On a system that can play the bottom octave, the mix will feel small.",
        buys="Headroom and translation. A light sub is loud, tidy and safe on small speakers.",
        right_when="The delivery target is small speakers, laptops and phones above everything else.",
    ),
    "low_end.subsonic_rumble": Tradeoff(
        costs=(
            "Level you can hear, spent on content you cannot. Energy below roughly 25 Hz moves "
            "your limiter without moving any listener."
        ),
        buys=(
            "Very little on most systems, though it is real on a large rig and it is part of "
            "the tail on some 808 designs."
        ),
        right_when=(
            "The track is cut for a sound system that reproduces below 30 Hz and the content "
            "down there is intentional rather than handling noise or a DC-adjacent offset."
        ),
    ),
    "low_end.section_collapse": Tradeoff(
        costs="The floor dropping out at the moment the listener is paying the most attention.",
        buys="Contrast. A section stripped of low end makes the one after it feel enormous.",
        right_when="You wrote a breakdown. Then this is the arrangement working, not failing.",
    ),
    "stereo_width.too_wide": Tradeoff(
        costs=(
            "Level and reliability. Width lives in the side channel, and anything summed to "
            "mono — a club rig, a phone speaker, a smart speaker — cancels part of it away."
        ),
        buys=(
            "Scale on headphones, which is where most people listen. A wide mix feels expensive "
            "and immersive in a way a narrow one never does."
        ),
        right_when=(
            "The headphone experience is the one you are optimising and the mono sum has been "
            "checked and is acceptable rather than assumed."
        ),
    ),
    "stereo_width.too_narrow": Tradeoff(
        costs="Room. With everything stacked in the centre, parts have to fight for the same space.",
        buys=(
            "Power and translation. A centred mix is the same mix everywhere, it survives mono, "
            "and it hits harder per dB because nothing is spent on the sides."
        ),
        right_when=(
            "The genre is centre-heavy by design — a lot of hip-hop and drill is deliberately "
            "narrow so the low end and the lead stay solid on any system."
        ),
    ),
    "vocal_balance.buried": Tradeoff(
        costs=(
            "Words. A listener who has to work to follow the lyric usually stops working and "
            "skips instead."
        ),
        buys=(
            "Space for whatever goes on top. A lead tucked under the bed is the correct choice "
            "on a beat, on a reference instrumental, and on any track where the topline has not "
            "been recorded yet."
        ),
        right_when=(
            "The lead you can hear is a hook or a sample meant to sit behind a performance that "
            "is not on this file. Then a forward vocal would be the mistake."
        ),
    ),
    "vocal_balance.too_loud": Tradeoff(
        costs=(
            "Cohesion. A lead sitting on top of the track rather than inside it makes the "
            "production behind it sound like a backing track."
        ),
        buys=(
            "Absolute intelligibility, and a lead that reads as the record on any speaker. "
            "Modern pop and country push the vocal harder than most people expect."
        ),
        right_when="The song is the vocal and the arrangement exists to hold it up.",
    ),
    "vocal_balance.inconsistent": Tradeoff(
        costs="Focus. A lead that drifts forward and back makes the listener keep re-adjusting.",
        buys="Performance. Dynamics in a vocal are expression, and riding them flat kills it.",
        right_when="The movement follows the delivery — a whispered line, a shouted one — rather than drifting at random.",
    ),
    "vocal_balance.topline_headroom": Tradeoff(
        costs=(
            "Nothing on its own. This is a note about how much space is left in the centre for "
            "a topline, not a fault in what is here."
        ),
        buys=(
            "A beat somebody can actually rap or sing over without carving the instrumental "
            "apart first, which is the entire job of a beat."
        ),
        right_when="Always, if this is a beat. That is what the measurement is for.",
    ),
}


#: hot / thin trade-offs for the per-band `frequency_balance.*` findings.
_BAND_TRADEOFF: Dict[str, Tuple[Tradeoff, Tradeoff]] = {
    # band slug -> (hot, thin)
    "sub": (
        Tradeoff(
            costs="Headroom, and boom on systems with any port resonance at all.",
            buys="Physical weight where it counts on a real system.",
            right_when="The music is built on the bottom octave and is mixed for systems that play it.",
        ),
        Tradeoff(
            costs="Scale. The mix will feel small anywhere the bottom octave is audible.",
            buys="Clean headroom and easy translation on phones and laptops.",
            right_when="The arrangement lives above 60 Hz and adding sub would only add mud.",
        ),
    ),
    "low_bass": (
        Tradeoff(
            costs="Definition. Too much 60-120 Hz turns kick and bass into one boom.",
            buys="Warmth and thickness, and a low end audible on speakers with no real sub.",
            right_when="The record is meant to feel heavy on ordinary speakers rather than deep on big ones.",
        ),
        Tradeoff(
            costs="Body. Without this band the bass exists as a note but not as a weight.",
            buys="Clarity through the low mids and a tighter, faster-sounding low end.",
            right_when="You have deliberately put the weight lower, in the sub, and want it to stay there.",
        ),
    ),
    "upper_bass": (
        Tradeoff(
            costs="A woolly, one-note bottom that masks the low mids above it.",
            buys="Fullness on small speakers, which cannot reproduce anything lower.",
            right_when="Phone and laptop playback is the priority and this band is carrying the bass line.",
        ),
        Tradeoff(
            costs="The bass disappearing entirely on anything without a woofer.",
            buys="A clean gap between the sub and the mids.",
            right_when="A separate sub layer is doing this job and doubling it would only blur things.",
        ),
    ),
    "low_mid": (
        Tradeoff(
            costs="The covered, boxy quality that makes a mix sound like it is behind a curtain.",
            buys="Warmth, closeness and body — a mix scooped here sounds thin and modern-cheap.",
            right_when="The aesthetic is warm and close: soul, lo-fi, vintage-leaning production.",
        ),
        Tradeoff(
            costs="Body. Vocals and guitars lose their chest and start to sound like a phone call.",
            buys="Clarity and separation, and space for the low end to be heard as low end.",
            right_when="The mix is dense and the low mids were the thing crowding it.",
        ),
    ),
    "mid": (
        Tradeoff(
            costs="A honky, mid-forward sound that gets tiring on any speaker with a mid driver.",
            buys="Presence on the smallest playback there is, where nothing else survives.",
            right_when="Phone-speaker playback matters more to you than the hi-fi listen.",
        ),
        Tradeoff(
            costs="Substance. A scooped midrange sounds impressive alone and vanishes in a playlist.",
            buys="A hi-fi, scooped 'smiley' curve that flatters big speakers.",
            right_when="You are deliberately chasing a scooped sound and have checked it on a laptop.",
        ),
    ),
    "upper_mid": (
        Tradeoff(
            costs="Fatigue. This is the most sensitive region of hearing and it gets tiring fastest.",
            buys="Attack and intelligibility — the band that makes snares crack and consonants land.",
            right_when="The track is meant to be aggressive and forward, and you know it is a choice.",
        ),
        Tradeoff(
            costs="Cut. Without it the mix sounds polite and gets lost between louder tracks.",
            buys="A smooth, easy listen that never becomes uncomfortable.",
            right_when="The music is meant to sit in the background rather than demand attention.",
        ),
    ),
    "presence": (
        Tradeoff(
            costs="Edge on consonants and cymbals, and the first thing to spit on lossy encoding.",
            buys="Detail and articulation — the band that makes a vocal sound close.",
            right_when="The lead needs to be right in the listener's face and you have checked it on earbuds.",
        ),
        Tradeoff(
            costs="Definition. The mix reads as dull and slightly distant.",
            buys="Comfort. Nothing here can ever become harsh.",
            right_when="You are after a soft, muted, deliberately unhyped top end.",
        ),
    ),
    "brilliance": (
        Tradeoff(
            costs="Hiss, and cymbals that sit in front of the music rather than behind it.",
            buys="Sparkle and a sense of expensive, modern polish.",
            right_when="The production is bright by design and the source material can support it.",
        ),
        Tradeoff(
            costs="Sheen. The top sounds rolled off and the mix reads as an old recording.",
            buys="A vintage, analogue-leaning character with no digital glare at all.",
            right_when="The rolled-off top is the aesthetic — lo-fi and tape-style production live here.",
        ),
    ),
    "air": (
        Tradeoff(
            costs="Nothing audible on most systems, but it is real headroom on a good one.",
            buys="Openness, and the sense of the mix extending past the speakers.",
            right_when="You added it on purpose with a shelf and it is doing what you wanted.",
        ),
        Tradeoff(
            costs="Openness. The mix sounds closed in rather than extending above the music.",
            buys="A darker, closer, more intimate sound with no fizz and no encoder artefacts.",
            right_when="The record is meant to sound close and dark, or the sources have no air to recover.",
        ),
    ),
}


_GENERIC_TRADEOFF = Tradeoff(
    costs=(
        "This is a measured distance from where releases in this genre sit. Whether it costs "
        "you anything depends on what you were going for."
    ),
    buys=(
        "Possibly the thing that makes the record yours. A departure from a reference is only "
        "a problem when nobody chose it."
    ),
    right_when="You made this call on purpose and it still sounds right on a second system.",
)


def _tradeoff(finding_id: str) -> Tradeoff:
    direct = _TRADEOFFS.get(finding_id)
    if direct is not None:
        return direct

    if finding_id.startswith("frequency_balance."):
        rest = finding_id.split(".", 1)[1]
        for suffix, index in (("_hot", 0), ("_thin", 1)):
            if rest.endswith(suffix):
                pair = _BAND_TRADEOFF.get(rest[: -len(suffix)])
                if pair is not None:
                    return pair[index]
    return _GENERIC_TRADEOFF


# ---------------------------------------------------------------------------
# Content assembly
# ---------------------------------------------------------------------------

_GRADE_WORD: Dict[str, str] = {
    "A": "release-ready", "B": "close", "C": "workable", "D": "needs a session", "F": "start again",
}

#: The technical grade is about defects and nothing else, so it needs its own
#: words. "Release-ready" off a composite that was mostly genre distance is how
#: a file with nothing wrong with it ended up graded D-; these say what the
#: number actually measured.
_TECHNICAL_WORD: Dict[str, str] = {
    "A": "nothing measurably wrong",
    "B": "small faults only",
    "C": "real faults to fix",
    "D": "significant faults",
    "F": "something is badly wrong",
}

_SEV_WORD: Dict[str, str] = {
    "critical": "critical", "major": "significant", "minor": "small", "clean": "clean",
}


def _dim_label(dimension: str) -> str:
    return DIMENSION_LABELS.get(dimension, dimension.replace("_", " ").title())


def _rank(finding: Finding) -> Tuple[int, float, float]:
    """Worst first: severity, then recoverable points, then confidence."""
    return (
        SEVERITY_RANK.get(finding.severity, 9),
        -_fin(finding.impact),
        -_fin(finding.confidence),
    )


def _confirmed(finding: Finding) -> bool:
    """Did the producer tell us this one was on purpose?

    `getattr` rather than attribute access throughout, because `/report` accepts
    a `MixAnalysis` posted by a client that predates the field entirely.
    """
    return bool(getattr(finding, "acknowledged", False))


def _open_question(finding: Finding):
    """The unanswered `Clarification` on this finding, or None.

    Answered means settled: the question stays on the finding so the document
    can show what was asked next to what was said, but it stops being an open
    question the moment it is acknowledged.
    """
    clar = getattr(finding, "clarification", None)
    if clar is None or _confirmed(finding):
        return None
    return clar


class _Numbering:
    """Sequential eyebrow numbers for the top-level sections.

    *Choices you confirmed* only exists on a report where the producer answered
    something, and a hard-coded "02 ·" on it would print a document that jumps
    from 01 to 03 on every report where they did not. The sections number
    themselves in the order `build_document` emits them instead.
    """

    def __init__(self) -> None:
        self._n = 0

    def __call__(self, label: str) -> str:
        self._n += 1
        return f"{self._n:02d} · {label}"


def _kind_of(finding: Finding) -> str:
    """Trust the payload's `kind`, but re-derive it when an old client omits it.

    `finding_kind` is a pure function of the id and is the single place the
    classification lives, so falling back to it costs nothing and means a
    document generated from a stale analysis still splits correctly.
    """
    kind = getattr(finding, "kind", None)
    return kind if kind in ("defect", "deviation") else finding_kind(finding.id)


def _places(value: float) -> int:
    """Decimal places that carry information rather than noise.

    `22.128 dB` implies a precision the measurement does not have; `0.166` on a
    0-1 index is exactly right. Scale the places to the magnitude.
    """
    v = abs(_fin(value))
    if v >= 100:
        return 0
    if v >= 10:
        return 1
    return 2 if v >= 1 else 3


def _evidence_target(ev: Evidence) -> str:
    unit = f" {ev.unit}" if ev.unit else ""
    if ev.target_range and len(ev.target_range) >= 2:
        nd = max(_places(ev.target_range[0]), _places(ev.target_range[1]))
        return _window(ev.target_range, ev.unit, nd)
    if ev.target is not None:
        return f"{_num(ev.target, _places(ev.target))}{unit}"
    return "—"


def _evidence_table(doc: Doc, finding: Finding, heading: str = "Measured") -> None:
    if not finding.evidence:
        return
    rows = []
    for ev in finding.evidence[:8]:
        rows.append([
            _clip(ev.label, 90),
            f"{_num(ev.value, _places(ev.value))}{(' ' + ev.unit) if ev.unit else ''}",
            _evidence_target(ev),
            _clip(ev.detail, 200) or "—",
        ])
    doc.h(4, heading)
    doc.table(["Measurement", "This mix", "Reference", "What it means"], rows, numeric=(1, 2))


def _moment_line(finding: Finding) -> str:
    """Every flagged timestamp, in playback order.

    Deliberately not "worst at" — the detector's own `detail` sentence already
    names the worst moment, and printing that again immediately underneath it
    reads like a bug. What this adds is the full set, ordered so you can work
    down the track rather than jump around it.
    """
    moments = finding.moments or []
    if not moments:
        return ""
    ordered = sorted(moments, key=lambda x: _fin(x.t_start))[:8]
    stamps = [f"{int(_fin(m.t_start) // 60)}:{_fin(m.t_start) % 60:04.1f}" for m in ordered]
    more = f", +{len(moments) - len(stamps)} more" if len(moments) > len(stamps) else ""
    return f"**Where to hear it:** {', '.join(stamps)}{more}."


def _first_action(finding_id: str, owned: Sequence[str]) -> str:
    """The complete first fix step, as the knowledge base authored it.

    `_resolved_steps` splits each action into a lead sentence and a spill so the
    fix list can bold the instruction and set the rest as body copy. That is the
    right shape for a numbered list and the wrong shape for *Start here*, which
    prints the lead sentence alone: several actions open with a pointer that
    means nothing on its own — "Find where it starts.", "Work out which half is
    wrong before reaching for anything." — and the sentence carrying the actual
    move is the one that got dropped.

    So this returns the action whole. Two short sentences is what the section
    promises and what a first move usually needs.
    """
    explainer = knowledge.explain(finding_id)
    if explainer is None:
        return ""
    steps = knowledge.applicable_steps(explainer, owned)
    if not steps:
        return ""
    return " ".join(str(steps[0].get("action", "")).split())


def _resolved_steps(finding_id: str, owned: Sequence[str]) -> List[Tuple[str, str]]:
    """Fix steps for this finding, resolved against the tools they actually own.

    `knowledge.applicable_steps` drops steps needing a capability they lack and
    substitutes the `without` variant where one exists, so nothing here tells
    somebody to reach for a plugin they do not have.
    """
    explainer = knowledge.explain(finding_id)
    if explainer is None:
        return []

    out: List[Tuple[str, str]] = []
    for step in knowledge.applicable_steps(explainer, owned):
        action, spill = _lead_sentence(step.get("action", ""))
        detail = " ".join(p for p in (spill, step.get("detail", "")) if p).strip()
        substituted = step.get("substituted_for")
        if substituted:
            article = "an" if substituted[:1].lower() in "aeiou" else "a"
            detail = (
                f"{detail} " if detail else ""
            ) + f"This is the route without {article} {substituted}, which you do not have."
        out.append((action, detail.strip()))
    return out


def _prescriptions(analysis: MixAnalysis) -> Dict[str, Prescription]:
    report = analysis.engineer
    if report is None:
        return {}
    return {p.finding_id: p for p in (report.prescriptions or []) if p.finding_id}


def _weave_prescription(doc: Doc, presc: Optional[Prescription]) -> None:
    """The AI layer's track-specific read, clearly attributed and never required."""
    if presc is None:
        return

    body: List[str] = []
    if presc.diagnosis:
        body.append(_clip(presc.diagnosis, 900))
    if presc.root_cause:
        body.append(f"**Root cause.** {_clip(presc.root_cause, 600)}")
    doc.note("ai", _clip(presc.headline or "On this track specifically", 160), body)

    if presc.moves:
        doc.h(4, "The moves, for this track")
        rows = []
        for move in sorted(presc.moves, key=lambda m: m.order)[:8]:
            settings = ", ".join(f"{k} {v}" for k, v in list(move.settings.items())[:6])
            rows.append([
                _clip(move.target, 60) or "—",
                _clip(move.action, 220),
                _clip(move.tool, 60) or "—",
                _clip(settings, 160) or "—",
            ])
        doc.table(["Where", "Do this", "Tool", "Settings"], rows)

    extras: List[str] = []
    if presc.alternative:
        extras.append(f"**Another route.** {_clip(presc.alternative, 400)}")
    if presc.do_not:
        extras.append(f"**Do not.** {_clip(presc.do_not, 400)}")
    doc.ul(extras)


# --- 1. cover --------------------------------------------------------------


def _plain_verdict(analysis: MixAnalysis, defects: List[Finding],
                   deviations: List[Finding], genre_label: str,
                   confirmed: Sequence[Finding] = ()) -> str:
    """One line, no jargon, that survives being the only thing anybody reads.

    `deviations` here is the open set — anything the producer has confirmed as
    deliberate is counted separately and described as a decision, because
    "there are ten places where this sits away from the reference" reads as ten
    outstanding jobs when two of them are finished conversations.
    """
    n_def, n_dev, n_ack = len(defects), len(deviations), len(confirmed)
    intent_is_beat = analysis.intent in ("beat", "instrumental")
    confirmed_tail = (
        f" {n_ack} further difference{'' if n_ack == 1 else 's'} "
        f"{'is' if n_ack == 1 else 'are'} on the record because you chose "
        f"{'it' if n_ack == 1 else 'them'}, and {'it is' if n_ack == 1 else 'they are'} "
        "not on any list here."
        if n_ack else ""
    )

    if n_def == 0 and n_dev == 0:
        return (
            "Nothing here is broken and nothing unresolved sits outside the "
            f"{genre_label} reference — this one is ready to go.{confirmed_tail}"
        )
    if n_def == 0:
        subject = "beat" if intent_is_beat else "mix"
        return (
            f"Nothing is broken. There {'is' if n_dev == 1 else 'are'} {n_dev} "
            f"place{'' if n_dev == 1 else 's'} where this {subject} sits away from the "
            f"{genre_label} reference, and some of {'that' if n_dev == 1 else 'those'} "
            f"may well be on purpose too.{confirmed_tail}"
        )
    # The title up to its colon: "Hard clipping: the waveform tops are squared
    # off" carries a full explanation the one-line verdict does not have room
    # for, and reads badly lower-cased mid-sentence.
    worst = defects[0].title.split(":", 1)[0].strip().lower()
    tail = (
        f", plus {n_dev} place{'' if n_dev == 1 else 's'} where it sits away from the "
        f"{genre_label} reference — some of which may be deliberate"
        if n_dev else ""
    )
    return (
        f"{n_def} thing{'' if n_def == 1 else 's'} here {'is' if n_def == 1 else 'are'} "
        f"genuinely wrong, starting with {worst}{tail}.{confirmed_tail}"
    )


def _score_entries(analysis: MixAnalysis) -> List[Tuple[str, str]]:
    """The cover's score lines, led by the ScoreCard where there is one.

    The single composite this replaces put a D- on a file with zero defects
    that was ready to master, and an A on one with two real defects, because it
    was dominated by distance from a genre profile rather than by whether
    anything was wrong. Two numbers fix that by refusing to answer two
    questions with one figure: *technical* is defects only and carries the
    grade, *reference match* is distance from the genre and deliberately
    carries none.

    Falls back to the composite for an analysis posted by a client from before
    `scores` existed — labelled as the composite, so nobody reads it as the
    verdict it could not support.
    """
    scores = getattr(analysis, "scores", None)
    if scores is None:
        grade = analysis.grade or "—"
        word = _GRADE_WORD.get(grade[:1].upper(), "")
        return [(
            "Score (composite)",
            f"{_num(analysis.health_score, 0)} / 100  ·  grade {grade}"
            + (f"  ·  {word}" if word else ""),
        )]

    grade = scores.technical_grade or "—"
    word = _TECHNICAL_WORD.get(grade[:1].upper(), "")
    return [
        ("Verdict", _clip(scores.headline, 160)),
        (
            "Technical",
            f"{_num(scores.technical, 0)} / 100  ·  grade {grade}"
            + (f"  ·  {word}" if word else "")
            + "  —  defects only, and the only graded number here",
        ),
        (
            "Reference match",
            f"{_num(scores.reference_match, 0)} / 100  ·  {_clip(scores.reference_label, 120)}"
            "  —  a description of distance from the genre, not a grade",
        ),
    ]


def _cover(doc: Doc, analysis: MixAnalysis, defects: List[Finding],
           deviations: List[Finding], confirmed: List[Finding],
           genre_label: str, generated_at: datetime) -> None:
    doc.h(1, "Mix report", "Mix Diagnostic")
    doc.p(analysis.filename or "Untitled mix", lead=True)

    found = (
        f"{len(defects)} defect{'' if len(defects) == 1 else 's'}, "
        f"{len(deviations)} open deviation{'' if len(deviations) == 1 else 's'}"
    )
    if confirmed:
        found += (
            f", {len(confirmed)} choice{'' if len(confirmed) == 1 else 's'} you confirmed"
        )

    # `ceiling_score` is the *composite's* ceiling, and next to "Technical
    # 100 / 100" an unqualified "Reachable score 84 / 100" reads as the file
    # getting worse for doing the work. It is only printed where the reader has
    # a composite to attach it to; with the split score it is the reference
    # match the fixes move, and that figure is not a target to hit.
    tail = (
        []
        if getattr(analysis, "scores", None) is not None
        else [("Reachable score",
               f"{_num(analysis.ceiling_score, 0)} / 100 if you do everything here")]
    )

    doc.add(Definitions(_score_entries(analysis) + [
        ("Genre reference", genre_label),
        ("Track type", TRACK_INTENT_LABELS.get(analysis.intent, analysis.intent)),
        ("Generated", generated_at.strftime("%d %B %Y, %H:%M UTC")),
        ("Findings", found),
    ] + tail))
    doc.p(_plain_verdict(analysis, defects, deviations, genre_label, confirmed), lead=True)

    if analysis.engineer and analysis.engineer.verdict:
        doc.note("ai", "The engineer's read", [_clip(analysis.engineer.verdict, 1200)])

    how_to_read = [
        "**Two numbers, not one.** *Technical* is the graded one and it counts defects — "
        "things that are wrong in any genre, for anyone, on any record. *Reference match* is "
        f"how close this sits to where {genre_label} releases usually sit, and it has no grade "
        "on purpose: a record can be excellent and a long way from its genre. A low reference "
        "match is a description of a track doing its own thing, never a mark against it.",
        "**Read *Start here* if you read nothing else.** *What's genuinely wrong* is the only "
        "section that is instructions; everything after it is information.",
    ]
    if confirmed:
        how_to_read.append(
            "**Choices you confirmed** collects the things you already told us were "
            "deliberate. They are in the document because they are part of what this record "
            "is, and they are explicitly not on any fix list."
        )
    how_to_read.append(
        f"**Where this differs from the {genre_label} reference** is a set of decisions, not "
        "faults. Each one says what it costs and what it buys, and any one the analysis "
        "cannot call for itself is written as an open question for you to answer."
    )
    how_to_read.append(
        "**The last two sections are reference material** — the concepts behind everything "
        "that fired, then every number with its unit and its target."
    )
    doc.note("note", "How to read this document", how_to_read)
    doc.add(PageBreak())


# --- 2. start here ---------------------------------------------------------


def _start_here(doc: Doc, analysis: MixAnalysis, ordered: List[Finding],
                owned: Sequence[str], prescriptions: Dict[str, Prescription],
                genre_label: str, num: _Numbering) -> None:
    doc.h(2, "Start here", num("The three that matter"))

    scores = getattr(analysis, "scores", None)
    if scores is not None:
        # The headline before the list, because "nothing here is broken" changes
        # what the three items underneath it mean. Without it a reader arrives
        # at three headings and assumes all three are damage.
        grade = scores.technical_grade or "—"
        doc.note(
            "verdict",
            _clip(scores.headline, 160),
            [
                f"**Technical {_num(scores.technical, 0)} / 100, grade {grade}** — that is "
                "defects only: things wrong in any genre, on any record, for anyone. "
                f"**Reference match {_num(scores.reference_match, 0)} / 100** — "
                f"{_clip(scores.reference_label, 120).rstrip('.')}, which is a description of "
                "how far this sits from the genre and carries no grade at all.",
            ],
        )

    if not ordered:
        doc.p(
            "Nothing is outstanding. Every dimension measured either inside the "
            f"{genre_label} reference, too close to it to be worth a note, or as something you "
            "have already confirmed was deliberate — so there is no highest-leverage move to "
            "name. The useful sections for you are *What's already working* and the appendix."
        )
        return

    doc.p(
        "Ranked by how much of the score each one is holding, defects before "
        "reference deviations. Each one says what it is, then the first move — in full, so "
        "you can act on it without reading any further.",
        lead=True,
    )

    if analysis.engineer and analysis.engineer.the_one_thing:
        doc.note("ai", "If you only do one thing",
                 [_clip(analysis.engineer.the_one_thing, 700)])

    for i, finding in enumerate(ordered[:3], 1):
        kind = _kind_of(finding)
        explainer = knowledge.explain(finding.id)
        kind_word = "Defect" if kind == "defect" else f"Differs from the {genre_label} reference"
        question = _open_question(finding)

        doc.h(3, f"{i}. {finding.title}")
        rows = [
            ("Kind", kind_word),
            ("Area", _dim_label(finding.dimension)),
            ("Severity", _SEV_WORD.get(finding.severity, finding.severity)),
            ("Worth", f"about {_num(finding.impact, 0)} points"),
        ]
        if question is not None:
            rows.append(("Open question", "Answer this before you touch it"))
        doc.add(Definitions(rows, compact=True))

        what = _sentence(explainer.headline) if explainer else _clip(finding.detail, 320)

        presc = prescriptions.get(finding.id)
        action = _first_action(finding.id, owned)
        if presc is not None and presc.moves:
            first = sorted(presc.moves, key=lambda m: m.order)[0]
            do = _sentence(f"First move: {first.action.rstrip('.')} on the {first.target}")
        elif action:
            do = _sentence(f"First move: {action[0].lower()}{action[1:]}")
        else:
            do = "The full write-up for this one is below."

        if question is None:
            doc.p(f"{what} {do}")
        else:
            # A first move stated flat here would be the analyser guessing
            # again — the same guess that called a deliberately thin intro the
            # dominant issue. The question comes first and the move is
            # conditional on the answer.
            doc.p(what)
            doc.note("question", _sentence(question.question), [
                f"**If yes, it was deliberate:** {_sentence(_clip(question.if_intended, 400))} "
                "Nothing to do.",
                f"**If no:** {_sentence(_clip(question.if_not, 400))} {do}",
            ])


# --- 3. choices you confirmed ----------------------------------------------


def _confirmed_section(doc: Doc, confirmed: List[Finding], num: _Numbering) -> None:
    """What the producer already told us was deliberate.

    Short on purpose. These are not findings any more and giving each one the
    full defect treatment — causes, fix steps, a stop condition — would put a
    fix list under a heading that says there is nothing to fix. What each entry
    owes the reader is the decision, the number behind it, and the trade it
    made, so that six weeks later the document still explains why the record
    sounds the way it does.
    """
    if not confirmed:
        return

    doc.add(PageBreak())
    doc.h(2, "Choices you confirmed", num("Decisions, not findings"))

    doc.p(
        "You were asked about each of these and said it was on purpose. That answer is the "
        "end of it: none of them counts against the technical score, none of them is on the "
        "session plan, and nothing later in this document asks you to change one. They are "
        "here because they are part of what this record is — and because the numbers behind "
        "them still explain a lot about how everything else measures.",
        lead=True,
    )

    for finding in confirmed:
        trade = _tradeoff(finding.id)
        question = getattr(finding, "clarification", None)

        doc.h(3, finding.title, _dim_label(finding.dimension))
        doc.add(Definitions([
            ("Status", "Confirmed deliberate — not something to fix"),
            ("Area", _dim_label(finding.dimension)),
            ("Finding id", finding.id),
        ], compact=True))

        if question is not None:
            doc.note("choice", _sentence(question.question), [
                f"**You said yes.** {_sentence(_clip(question.if_intended, 500))}",
            ])

        doc.p(_sentence(finding.detail))

        doc.h(4, "What it buys you")
        doc.p(trade.buys)

        doc.h(4, "What it costs, so you know where to listen")
        doc.p(
            f"{trade.costs} None of that is a reason to undo it — you have already weighed it "
            "— but it is where to check first if the track ever behaves oddly on a system you "
            "have not tried."
        )

        _evidence_table(doc, finding, heading="The numbers behind it")

    doc.note("note", "Changing your mind is allowed", [
        "If one of these turns out not to have been deliberate after all, answer the question "
        "the other way in the app and re-run this document. The finding comes back with its "
        "full write-up, its fix steps and its score impact, exactly as it would have been.",
    ])


# --- 4. defects ------------------------------------------------------------


def _defect_section(doc: Doc, defects: List[Finding], owned: Sequence[str],
                    prescriptions: Dict[str, Prescription], num: _Numbering) -> None:
    doc.add(PageBreak())
    doc.h(2, "What's genuinely wrong", num("Defects"))

    if not defects:
        doc.note("good", "No defects found", [
            "Nothing in this track is broken in a way that is wrong regardless of genre, "
            "intent or taste — no clipping, no inter-sample overs, no inverted polarity, no "
            "mono cancellation, no limiter distortion. Everything the analysis flagged is a "
            "difference from a reference, and those are in the next section.",
        ])
        return

    doc.p(
        "A defect is wrong no matter what the record is trying to be. Nobody chooses a "
        "squared-off waveform, an inverted channel or a mix that vanishes in mono, and no "
        "genre wants one. These are the only findings in this document you should treat as "
        "instructions rather than as information.",
        lead=True,
    )

    for finding in defects:
        _finding_body(doc, finding, owned, prescriptions, is_defect=True)


# --- 5. deviations ---------------------------------------------------------


def _deviation_section(doc: Doc, deviations: List[Finding], owned: Sequence[str],
                       prescriptions: Dict[str, Prescription], analysis: MixAnalysis,
                       genre_label: str, num: _Numbering,
                       confirmed: Sequence[Finding] = ()) -> None:
    doc.add(PageBreak())
    doc.h(2, f"Where this differs from the {genre_label} reference", num("Deviations"))

    n_ack = len(confirmed)
    if not deviations:
        doc.p(
            f"Nothing unresolved measured outside the {genre_label} reference windows."
            + (
                f" The {n_ack} difference{'' if n_ack == 1 else 's'} the analysis did find "
                f"{'is' if n_ack == 1 else 'are'} in *Choices you confirmed*, where you told "
                f"us {'it was' if n_ack == 1 else 'they were'} deliberate."
                if n_ack else
                " That is unusual and worth noting: this track reads as its genre on every "
                "dimension the analysis compares."
            )
        )
        return

    doc.p(
        f"Everything below is a measured difference between this track and where {genre_label} "
        "releases actually sit. That is all it is. The reference is a description of a genre, "
        "not a specification for a record, and the departures are frequently the reason a "
        "track sounds like itself rather than like everything else.",
        lead=True,
    )
    doc.p(
        "So each one gets three things: what it costs, what it buys, and the case in which it "
        "is the right call. Read them as decisions to confirm or reverse, not as a list of "
        "faults. If a departure was deliberate and it still sounds right on a second system, "
        "the correct action is to leave it alone."
    )

    if n_ack:
        doc.p(
            f"{n_ack} further difference{'' if n_ack == 1 else 's'} "
            f"{'is' if n_ack == 1 else 'are'} not in this section at all: you have already "
            f"confirmed {'it was' if n_ack == 1 else 'they were'} deliberate, so "
            f"{'it lives' if n_ack == 1 else 'they live'} in *Choices you confirmed* instead."
        )

    if any(_open_question(f) is not None for f in deviations):
        doc.note("note", "Some of these are questions, not conclusions", [
            "Where the analysis could not tell a decision from a mistake, it says so and asks "
            "rather than guessing. A quiet intro measures identically whether you wrote it "
            "that way or the sub came in late, and calling that a problem on a track where it "
            "was the arrangement is exactly the kind of thing that makes a report worth "
            "ignoring.",
            "Each of those entries carries a boxed question with both answers written out. "
            "Answer it in the app and the report re-scores itself: a yes moves that entry into "
            "*Choices you confirmed* and takes it off every list here.",
        ])

    if analysis.intent in ("beat", "instrumental"):
        doc.note("note", "This was analysed as a beat", [
            "A beat is measured differently from a finished song. A lead tucked under the bed "
            "is a production decision rather than a balance problem, an open midrange is space "
            "somebody else is going to fill, and bursty high-frequency content is read as hats "
            "and shakers rather than as vocal sibilance. The reference windows below still come "
            f"from {genre_label}, but that is where they stop applying.",
        ])

    for finding in deviations:
        _finding_body(doc, finding, owned, prescriptions, is_defect=False)


# --- shared finding body ---------------------------------------------------


def _finding_body(doc: Doc, finding: Finding, owned: Sequence[str],
                  prescriptions: Dict[str, Prescription], is_defect: bool) -> None:
    explainer = knowledge.explain(finding.id)
    presc = prescriptions.get(finding.id)

    question = _open_question(finding)

    doc.h(3, finding.title, _dim_label(finding.dimension))
    points = _num(finding.impact, 0)
    rows = [
        ("Severity", _SEV_WORD.get(finding.severity, finding.severity)),
        ("Confidence", _pct(finding.confidence)),
        ("Score impact", f"{points} {'pt' if points == '1' else 'pts'}"),
        ("Finding id", finding.id),
    ]
    if question is not None:
        rows.insert(0, ("Status", "Open question — see the box below"))
    doc.add(Definitions(rows, compact=True))

    if explainer:
        doc.note(
            "defect" if is_defect else "reference",
            _sentence(explainer.headline),
        )

    doc.p(_sentence(finding.detail))
    moment = _moment_line(finding)
    if moment:
        doc.p(moment)

    if question is not None:
        # Worded so the document alone is enough to decide with. Somebody
        # reading a printout on a train has no app in front of them, and "there
        # is an unanswered question here" without the two answers written out is
        # a dead end rather than a prompt.
        doc.note("question", f"Decide this first: {_sentence(question.question)}", [
            _sentence(_clip(question.context, 600)),
            f"**If it was deliberate —** {_sentence(_clip(question.if_intended, 500))} "
            "Everything below this box is then background rather than instruction: it is "
            "worth knowing what the choice costs, and there is nothing here you need to do.",
            f"**If it was not —** {_sentence(_clip(question.if_not, 500))} That is what the "
            "steps below are for.",
            "Answering this in the app re-scores the report and moves this entry into "
            "*Choices you confirmed*. Nothing about the audio changes either way — only what "
            "the report makes of it.",
        ])

    if explainer:
        doc.h(4, "What this actually is")
        for para in _paragraphs(explainer.what_it_is):
            doc.p(para)

        doc.h(4, "What you hear")
        for para in _paragraphs(explainer.what_you_hear):
            doc.p(para)

        doc.h(4, "What it costs" if not is_defect else "Why it matters")
        for para in _paragraphs(explainer.why_it_matters):
            doc.p(para)

    if not is_defect:
        trade = _tradeoff(finding.id)
        doc.h(4, "What it buys")
        doc.p(trade.buys)
        if trade.right_when:
            doc.note("note", "This is the right call when", [trade.right_when])

    _evidence_table(doc, finding)

    if explainer and explainer.common_causes:
        doc.h(4, "Usually caused by")
        doc.ul(explainer.common_causes)

    steps = _resolved_steps(finding.id, owned)
    if steps:
        if is_defect:
            heading = "How to fix it"
        elif question is not None:
            heading = "How to change it, if the answer above was no"
        else:
            heading = "How to change it, if you want to"
        doc.h(4, heading)
        doc.steps(steps)

    _weave_prescription(doc, presc)

    stop = []
    if explainer and explainer.how_to_verify:
        stop.append(explainer.how_to_verify)
    if presc is not None and presc.done_when:
        stop.append(f"**On this track:** {_clip(presc.done_when, 400)}")
    if stop:
        doc.note("good", "You are done when", stop)


def _paragraphs(text: str) -> List[str]:
    return [p.strip() for p in str(text or "").split("\n\n") if p.strip()]


# --- 5. what's working -----------------------------------------------------


def _working_section(doc: Doc, analysis: MixAnalysis, genre_label: str,
                     num: _Numbering) -> None:
    doc.add(PageBreak())
    doc.h(2, "What's already working", num("Do not touch these"))

    clean = [d for d in (analysis.dimensions or []) if d.severity == "clean"]
    doc.p(
        "The most expensive mistake after a report like this is fixing something that was "
        "already right. These dimensions measured inside the "
        f"{genre_label} reference, and the headline on each says what the number actually was — "
        "so if a move you make later pushes one of them out, you will know which one.",
        lead=True,
    )

    if not clean:
        doc.p(
            "Nothing scored clean this time. That is not a disaster — most of what fired is in "
            "the deviations section and may be deliberate — but there is no dimension here that "
            "is safely out of scope for the session."
        )
    else:
        doc.table(
            ["Dimension", "Score", "What was measured"],
            [[_dim_label(d.dimension), _num(d.score, 0), _clip(d.headline, 400)] for d in clean],
            numeric=(1,),
        )

    if analysis.engineer and analysis.engineer.strengths:
        doc.note("ai", "What the engineer singled out",
                 [_clip(s, 400) for s in analysis.engineer.strengths[:6]])


# --- 6. session plan -------------------------------------------------------


def _session_plan(doc: Doc, ordered: List[Finding], owned: Sequence[str],
                  prescriptions: Dict[str, Prescription], analysis: MixAnalysis,
                  num: _Numbering) -> None:
    doc.add(PageBreak())
    doc.h(2, "Your session plan", num("In this order"))

    if not ordered:
        doc.p("Nothing to schedule. Bounce it and move on.")
        return

    doc.p(
        "Defects first, because a deviation measured on top of a clipped render is measuring "
        "the clipping. Then the reference deviations, worst first — and every one of those is "
        "optional. Times are for the move itself, not for the listening either side of it.",
        lead=True,
    )

    items: List[Tuple[str, str]] = []
    total = 0
    for finding in ordered:
        explainer = knowledge.explain(finding.id)
        presc = prescriptions.get(finding.id)
        minutes = int(_fin(presc.minutes if presc else 0)) or (explainer.minutes if explainer else 10)
        total += minutes

        steps = _resolved_steps(finding.id, owned)
        lead = _clip(steps[0][0], 180) if steps else _clip(finding.title, 120)
        kind = _kind_of(finding)
        prefix = "Fix" if kind == "defect" else "Decide"
        if kind == "defect":
            tag = ""
        elif _open_question(finding) is not None:
            tag = " *(only if you answered no to the question on this one)*"
        else:
            tag = " *(optional — confirm it was a choice first)*"
        items.append((
            f"**{prefix}: {_dim_label(finding.dimension)}.** {_sentence(lead)}{tag}",
            _minutes(minutes),
        ))

    items.append((
        "**Bounce it.** Render a fresh file rather than trusting the in-session meters — "
        "clipping and true-peak overs only exist in the render.",
        "5 min",
    ))
    items.append((
        "**Re-analyse the bounce and compare.** A revision that measures better is the only "
        "proof the moves landed.",
        "5 min",
    ))
    total += 10

    doc.add(Checklist(items))
    scores = getattr(analysis, "scores", None)
    if scores is None:
        doc.p(f"**Estimated total: {_minutes(total)}.** Composite score now "
              f"{_num(analysis.health_score, 0)}, reachable {_num(analysis.ceiling_score, 0)} "
              "with everything above applied.")
    else:
        doc.p(
            f"**Estimated total: {_minutes(total)}.** Technical score now "
            f"{_num(scores.technical, 0)} / 100, grade {scores.technical_grade} — and only the "
            "defects on this list move it, because that is the only thing it counts. The "
            "reference-match figure moves with the deviations, but moving it is not "
            "automatically an improvement: it measures how much this sounds like the rest of "
            f"the genre. (The older composite reads {_num(analysis.health_score, 0)} now and "
            f"{_num(analysis.ceiling_score, 0)} with everything above applied.)"
        )

    if analysis.engineer and analysis.engineer.session_plan:
        doc.note("ai", "The engineer's ordering",
                 [_clip(s, 300) for s in analysis.engineer.session_plan[:10]])

    if analysis.mastering_blockers:
        doc.h(3, "Before this goes to mastering")
        doc.ul([_clip(b, 400) for b in analysis.mastering_blockers[:8]])
    elif analysis.mastering_ready:
        doc.note("good", "Mastering-ready", [
            "Nothing in the measurements blocks a mastering engineer from working on this.",
        ])


# --- 7. the concepts -------------------------------------------------------


def _concepts_section(doc: Doc, ordered: List[Finding], genre_label: str,
                      num: _Numbering) -> None:
    doc.add(PageBreak())
    doc.h(2, "The concepts behind this", num("A short course from your own track"))

    entries: List[Tuple[str, str, str]] = []
    seen: set = set()
    for finding in ordered:
        explainer = knowledge.explain(finding.id)
        if explainer is None or not explainer.learn_more:
            continue
        key = explainer.learn_more[:120]
        if key in seen:
            continue
        seen.add(key)
        entries.append((_dim_label(finding.dimension), explainer.headline,
                        explainer.learn_more))

    if not entries:
        doc.p(
            "Nothing fired, so there is no theory to attach to it. The reference windows for "
            f"{genre_label} are in the appendix if you want to see what this track was measured "
            "against."
        )
        return

    doc.p(
        "Every explanation below is here because something in *your* track triggered it — this "
        "is not a general primer. Each one is the underlying idea rather than the fix: why the "
        "problem exists at all, which is the part that transfers to the next record.",
        lead=True,
    )

    for area, title, learn in entries:
        # Lead the heading with the area, not the symptom. This section is
        # reference material somebody comes back to weeks later — "Mud &
        # Low-Mid Buildup" is findable, "Your mix sounds covered" is a
        # complaint about one track that happens to be filed here. The symptom
        # still follows it, so the heading also maps back to the finding above.
        # A heading is not a sentence, so the explainer's terminal full stop
        # goes. Most headlines are "symptom — elaboration"; the elaboration is
        # already the first thing the body says, and keeping it here produces a
        # heading with two dashes in it that reads like a run-on.
        symptom = " ".join(str(title).split()).rstrip(".").split(" — ")[0]
        doc.h(3, f"{area} — {symptom}")
        for para in _paragraphs(learn):
            doc.p(para)


# --- 8. appendix -----------------------------------------------------------


def _measurement_rows(m: Measurements, genre: str) -> List[Tuple[str, List[List[str]]]]:
    """(group, rows) for the appendix. Rows are [metric, value, unit, target, note]."""
    p = targets.get_profile(genre)
    ld, cl, sp, st, ph = m.loudness, m.clipping, m.spectral, m.stereo, m.phase
    dy, tr, le, vo, cy = m.dynamics, m.transients, m.low_end, m.vocal, m.clarity

    def row(label, value, unit, target, note=""):
        return [label, value, unit, target, note]

    groups: List[Tuple[str, List[List[str]]]] = []

    groups.append(("File", [
        row("Duration", _num(m.duration_seconds, 1), "s", "—"),
        row("Sample rate", _num(m.sample_rate, 0), "Hz", "—",
            f"decoded from {_num(m.original_sample_rate, 0)} Hz"),
        row("Channels", "mono" if m.is_mono else "stereo", "", "stereo",
            "Width, phase and vocal balance need two channels" if m.is_mono else ""),
        row("Bit depth", _num(m.bit_depth, 0) if m.bit_depth else "—", "bit", "—"),
    ]))

    groups.append(("Loudness and delivery", [
        row("Integrated loudness", _num(ld.integrated_lufs, 1), "LUFS",
            _window(p.integrated_lufs, "LUFS"), "Average loudness over the whole track"),
        row("Short-term max", _num(ld.short_term_max_lufs, 1), "LUFS", "—", "Loudest 3 s window"),
        row("Momentary max", _num(ld.momentary_max_lufs, 1), "LUFS", "—", "Loudest 400 ms window"),
        row("Loudness range", _num(ld.loudness_range_lu, 1), "LU",
            _window(p.loudness_range_lu, "LU"), "Spread between quiet and loud sections"),
        row("True peak", _num(ld.true_peak_dbtp, 2), "dBTP", "≤ -1.0 dBTP",
            "The level after a converter reconstructs the waveform"),
        row("Sample peak", _num(ld.sample_peak_dbfs, 2), "dBFS", "≤ -1.0 dBFS",
            "The highest stored sample"),
        row("Peak to loudness (PLR)", _num(ld.plr_db, 1), "dB", "—",
            "True peak minus integrated; headroom above the average"),
        row("PSR, 10th percentile", _num(ld.psr_p10_db, 1), "dB",
            _window(p.psr_p10_db, "dB"), "Peak above short-term loudness in the densest moments"),
        row("PSR, median", _num(ld.psr_median_db, 1), "dB", "—"),
    ]))

    groups.append(("Clipping", [
        row("Clipped samples", _num(cl.clipped_samples, 0), "samples", "0"),
        row("Clipped share", _num(cl.clip_percentage, 4), "%", "0"),
        row("Longest flat run", _num(cl.longest_flat_run, 0), "samples", "0",
            "Consecutive pinned samples — a plateau rather than a peak"),
        row("Flat-topped runs", _num(cl.flat_run_count, 0), "runs", "0"),
        row("Inter-sample overs", _num(cl.inter_sample_overs, 0), "overs", "0",
            "Peaks that only exist once the waveform is reconstructed"),
        row("Distortion index", _num(cl.distortion_index, 3), "", "≤ 0.16",
            "High-frequency residue consistent with clipping"),
        row("Float over unity", _yes(cl.is_float_over_unity), "", "no"),
    ]))

    groups.append(("Dynamics", [
        row("Crest factor", _num(dy.crest_factor_db, 1), "dB", _window(p.crest_factor_db, "dB"),
            "Peak above RMS across the track"),
        row("Micro-dynamics", _num(dy.micro_dynamics_db, 1), "dB", _window(p.micro_dynamics_db, "dB"),
            "Crest inside each 50 ms frame — survives limiting or does not"),
        row("Macro-dynamics", _num(dy.macro_dynamics_lu, 1), "LU", "—", "Section-to-section variation"),
        row("TT-DR value", _num(dy.dr_value, 1), "", "—", "The DR meter figure"),
        row("RMS level", _num(dy.rms_db, 1), "dBFS", "—"),
        row("Pumping index", _num(dy.pumping_index, 3), "", "≤ 0.5",
            f"Envelope modulation locked to {_num(dy.pumping_rate_hz, 2)} Hz"),
        row("Gain reduction estimate", _num(dy.gain_reduction_estimate_db, 1), "dB", "—"),
    ]))

    groups.append(("Spectral balance", [
        row("Spectral tilt", _num(sp.spectral_tilt_db_per_decade, 1), "dB/decade", "—",
            "How fast energy falls as frequency rises"),
        row("Spectral centroid", _num(sp.spectral_centroid_hz, 0), "Hz", "—",
            "The balance point of the spectrum"),
        row("Mud ratio", _num(sp.mud_ratio_db, 1), "dB", _window(p.mud_ratio_db, "dB"),
            "150-400 Hz against 60-120 Hz"),
        row("Low-mid to mid", _num(sp.mud_to_mid_db, 1), "dB", "—", "150-400 Hz against 1-3 kHz"),
        row("Boxiness", _num(sp.boxiness_db, 1), "dB", "—"),
        row("Harshness index", _num(sp.harshness_index, 3), "", f"≤ {_num(p.harshness_max, 2)}"),
        row("Sibilance index", _num(sp.sibilance_index, 3), "", f"≤ {_num(p.sibilance_max, 2)}",
            "Burstiness in 5-9 kHz. Only read as a vocal issue when a vocal is present"),
        row("Sharpness", _num(sp.sharpness_acum, 2), "acum", f"≤ {_num(p.sharpness_max_acum, 2)}",
            "Psychoacoustic sharpness, Zwicker"),
        row("Resonances found", _num(len(sp.resonances or []), 0), "", "—"),
    ]))

    groups.append(("Stereo and phase", [
        row("Correlation", _num(st.correlation, 2), "", f"≥ {_num(p.correlation_min, 2)}",
            "+1 is mono, 0 is uncorrelated, -1 cancels in mono"),
        row("Width (Side/Mid)", _num(st.width, 2), "", _window(p.stereo_width, "", 2),
            "How much of the signal is in the sides"),
        row("Mono sum loss", _num(st.mono_sum_loss_db, 2), "dB", "≥ -1.0 dB",
            "Level lost when the mix is summed to mono"),
        row("Low-end side energy", _num(st.low_end_side_energy_db, 1), "dB", "—",
            "Side content below 120 Hz"),
        row("Low-end mono ratio", _num(le.low_end_mono_ratio, 2), "", f"≥ {_num(p.low_end_mono_min, 2)}"),
        row("L/R balance", _signed(st.balance_db), "dB", "0 ± 1",
            "Positive is right-heavy"),
        row("Polarity inverted", _yes(ph.polarity_inverted), "", "no"),
        row("Mono compatible", _yes(ph.mono_compatible), "", "yes"),
        row("Worst band correlation", _num(ph.worst_band_correlation, 2), "", "—",
            f"in the {ph.worst_band} band" if ph.worst_band else ""),
        row("Mono source", _yes(st.is_mono_source), "", "—"),
    ]))

    groups.append(("Low end", [
        row("Kick detected", _yes(le.kick_detected), "", "—", f"{_num(le.kick_count, 0)} hits"),
        row("Kick fundamental", _num(le.kick_fundamental_hz, 1), "Hz", "—"),
        row("Bass fundamental", _num(le.bass_fundamental_hz, 1), "Hz", "—"),
        row("Sub energy", _num(le.sub_energy_db, 1), "dB", _window(p.sub_energy_db, "dB")),
        row("Kick/bass collision", _num(le.kick_bass_collision_db, 1), "dB",
            f"≤ {_num(p.kick_bass_collision_max_db, 1)} dB", "Shared energy in the overlap region"),
        row("Ducking depth", _num(le.ducking_depth_db, 1), "dB", "—",
            f"sidechain {'detected' if le.has_sidechain else 'not detected'}"),
        row("Kick definition", _num(le.kick_definition_db, 1), "dB", "—",
            "Kick transient above the surrounding low end"),
        row("Sub rumble", _num(le.sub_rumble_db, 1), "dB", "—", "Energy below 25 Hz"),
    ]))

    groups.append(("Transients", [
        row("Punch index", _num(tr.punch_index, 3), "", f"≥ {_num(p.punch_min, 2)}",
            "Transient peak against the 50 ms sustain behind it"),
        row("Transient to sustain", _num(tr.transient_to_sustain_db, 1), "dB", "—"),
        row("Attack time", _num(tr.attack_time_ms, 1), "ms", "—"),
        row("Smearing index", _num(tr.smearing_index, 3), "", "—"),
        row("Onset density", _num(tr.onset_density, 2), "/s", "—"),
        row("Estimated tempo", _num(tr.estimated_tempo, 1), "BPM", "—"),
    ]))

    groups.append(("Lead / centre channel", [
        row("Lead detected", _yes(vo.vocal_present), "", "yes" if p.vocal_expected else "—",
            f"confidence {_pct(vo.vocal_confidence)}"),
        row("Prominence", vo.vocal_prominence, "", "—",
            "absent / tucked / balanced / forward"),
        row("Lead to instruments", _num(vo.vocal_to_instrument_db, 1), "dB",
            _window(p.vocal_to_instrument_db, "dB")),
        row("Centre energy ratio", _num(vo.center_energy_ratio, 3), "", "—"),
        row("Intelligibility", _num(vo.intelligibility_index, 3), "", "—"),
        row("Presence balance", _num(vo.presence_balance_db, 1), "dB", "—"),
        row("Sibilance level", _num(vo.sibilance_db, 1), "dB", "—"),
        row("Level consistency", _num(vo.consistency_db, 1), "dB", "—",
            "Spread of lead level over time; high is uneven"),
    ]))

    groups.append(("Clarity", [
        row("Clarity index", _num(cy.clarity_index, 3), "", f"≥ {_num(p.clarity_min, 2)}"),
        row("Masking index", _num(cy.masking_index, 3), "", f"≤ {_num(p.masking_max, 2)}"),
        row("Spectral flatness", _num(cy.spectral_flatness, 4), "", "—"),
        row("Spectral contrast", _num(cy.spectral_contrast, 2), "", "—"),
        row("Definition", _num(cy.definition_db, 1), "dB", "—"),
        row("Worst congested band", cy.worst_congested_band or "—", "", "—"),
    ]))

    return groups


def _appendix(doc: Doc, analysis: MixAnalysis, genre_label: str, num: _Numbering) -> None:
    doc.add(PageBreak())
    doc.h(2, "Appendix: every measurement", num("The receipts"))

    m = analysis.measurements
    doc.p(
        "Nothing on the previous pages is a judgement call about how something sounds — every "
        "one of them comes from a number in this appendix. The reference column is where "
        f"{genre_label} releases sit, not a pass mark: a value outside it is a difference, and "
        "the defect and deviation sections are where that difference is interpreted.",
        lead=True,
    )

    for group, rows in _measurement_rows(m, analysis.genre):
        doc.h(3, group)
        doc.table(["Measurement", "Value", "Unit", f"{genre_label} reference", "What it is"],
                  rows, numeric=(1,))

    bands = (m.spectral.bands if m.spectral else None) or []
    if bands:
        doc.h(3, "Frequency bands")
        doc.p(
            "Each band's level against the "
            f"{genre_label} target curve. Levels are relative to the mix's own broadband level, "
            "so this is balance rather than absolute loudness — a whole mix turned up does not "
            "move these numbers."
        )
        doc.table(
            ["Band", "Range", "Level", "Target", "Difference", "Verdict"],
            [[
                b.name.replace("_", " ").title(),
                f"{_num(b.low_hz, 0)}-{_num(b.high_hz, 0)} Hz",
                f"{_num(b.level_db, 1)} dB",
                f"{_num(b.target_db, 1)} dB",
                f"{_signed(b.deviation_db)} dB",
                b.verdict,
            ] for b in bands],
            numeric=(2, 3, 4),
        )

    if analysis.platform_targets:
        doc.h(3, "Delivery targets")
        doc.p(
            "Where this master lands on each platform after its own normalisation. "
            + doc.term("Normalisation", "a fixed volume change the platform applies so every "
                       "track in a playlist arrives at a similar loudness")
            + " is a level change and nothing else — it does not compress and it cannot undo "
              "anything spent getting there."
        )
        doc.table(
            ["Platform", "Target", "Delta", "Turned down", "Peak OK", "Verdict"],
            [[
                t.platform,
                f"{_num(t.target_lufs, 1)} LUFS",
                f"{_signed(t.delta_lufs)} LU",
                _yes(t.will_be_turned_down),
                _yes(t.peak_ok),
                t.verdict,
            ] for t in analysis.platform_targets],
            numeric=(1, 2),
        )

    sections = m.sections
    if sections and sections.available and sections.sections:
        doc.h(3, "Sections")
        doc.table(
            ["#", "Section", "Start", "End", "Loudness", "Crest", "Width"],
            [[
                str(s.index), s.label,
                f"{int(s.t_start // 60)}:{s.t_start % 60:04.1f}",
                f"{int(s.t_end // 60)}:{s.t_end % 60:04.1f}",
                f"{_num(s.integrated_lufs, 1)} LUFS",
                f"{_num(s.crest_factor_db, 1)} dB",
                _num(s.stereo_width, 2),
            ] for s in sections.sections],
            numeric=(4, 5, 6),
        )

    stems = m.stems
    if stems and stems.available and stems.stems:
        doc.h(3, "Separated sources")
        doc.p(f"Measured after source separation with {stems.model_name or 'the separation model'}.")
        doc.table(
            ["Source", "Level", "vs mix", "Peak", "Crest", "Width", "Punch"],
            [[
                s.kind.title(),
                f"{_num(s.level_lufs, 1)} LUFS",
                f"{_signed(s.level_ratio_db)} dB",
                f"{_num(s.peak_dbfs, 1)} dBFS",
                f"{_num(s.crest_factor_db, 1)} dB",
                _num(s.stereo_width, 2),
                _num(s.transient_punch, 2),
            ] for s in stems.stems if s.present],
            numeric=(1, 2, 3, 4, 5, 6),
        )

    ref = analysis.reference
    if ref is not None:
        doc.h(3, "Against your reference track")
        doc.table(
            ["Measurement", "Difference"],
            [
                ["Integrated loudness", f"{_signed(ref.integrated_lufs)} LU"],
                ["Dynamic range", f"{_signed(ref.dynamic_range_db)} dB"],
                ["Stereo width", _signed(ref.stereo_width, 2)],
                ["True peak", f"{_signed(ref.true_peak_dbtp)} dB"],
                ["Similarity", f"{_num(ref.similarity, 0)} / 100"],
            ],
            numeric=(1,),
        )
        if ref.biggest_gaps:
            doc.ul(ref.biggest_gaps[:8])

    _reference_profile(doc, analysis.genre, genre_label)

    if analysis.warnings:
        doc.h(3, "Analyser notes")
        doc.ul([_clip(w, 400) for w in analysis.warnings[:10]])


def _reference_profile(doc: Doc, genre: str, genre_label: str) -> None:
    p = targets.get_profile(genre)
    doc.h(3, f"The {genre_label} reference itself")
    doc.p(
        "For completeness: the windows every comparison in this document was made against. "
        "These are derived from where commercial releases in the genre actually measure, not "
        "from a standard — which is why a track can sit outside one of them and be entirely "
        "correct."
    )
    if p.notes:
        doc.p(f"*{p.notes}*")
    doc.table(
        ["Reference window", "Range"],
        [
            ["Integrated loudness", _window(p.integrated_lufs, "LUFS")],
            ["Loudness range", _window(p.loudness_range_lu, "LU")],
            ["Crest factor", _window(p.crest_factor_db, "dB")],
            ["Micro-dynamics", _window(p.micro_dynamics_db, "dB")],
            ["PSR (10th percentile)", _window(p.psr_p10_db, "dB")],
            ["Stereo width", _window(p.stereo_width, "", 2)],
            ["Correlation floor", _num(p.correlation_min, 2)],
            ["Low-end mono floor", _num(p.low_end_mono_min, 2)],
            ["Mud ratio", _window(p.mud_ratio_db, "dB")],
            ["Sub energy", _window(p.sub_energy_db, "dB")],
            ["Harshness ceiling", _num(p.harshness_max, 2)],
            ["Sibilance ceiling", _num(p.sibilance_max, 2)],
            ["Punch floor", _num(p.punch_min, 2)],
            ["Clarity floor", _num(p.clarity_min, 2)],
            ["Masking ceiling", _num(p.masking_max, 2)],
            ["Lead vs instruments", _window(p.vocal_to_instrument_db, "dB")],
        ],
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _sanitise(analysis: MixAnalysis) -> None:
    """Bound the client-supplied prose in place.

    The document renders everything through `_inline`, which escapes before it
    marks up, so this is not an injection defence — it is a size defence. The
    payload arrives from the browser and an unbounded string there is an
    unbounded document here. The limits are far above anything the analyser
    itself produces, so real content is never touched.
    """
    analysis.filename = _clip(analysis.filename, 200)
    analysis.genre = _clip(analysis.genre, 60)
    analysis.mastering_blockers = [_clip(b, 600) for b in (analysis.mastering_blockers or [])[:12]]
    analysis.warnings = [_clip(w, 600) for w in (analysis.warnings or [])[:12]]

    for finding in analysis.findings or []:
        finding.title = _clip(finding.title, 200)
        finding.detail = _clip(finding.detail, 2000)
        finding.evidence = list(finding.evidence or [])[:12]
        for ev in finding.evidence:
            ev.label = _clip(ev.label, 120)
            ev.detail = _clip(ev.detail, 300)
        # The question is rendered into the document, so it is client-supplied
        # prose like everything else here and gets the same size bound.
        clar = getattr(finding, "clarification", None)
        if clar is not None:
            clar.question = _clip(clar.question, 400)
            clar.context = _clip(clar.context, 1200)
            clar.if_intended = _clip(clar.if_intended, 1200)
            clar.if_not = _clip(clar.if_not, 1200)
    for dim in analysis.dimensions or []:
        dim.headline = _clip(dim.headline, 600)


def build_document(
    analysis: MixAnalysis,
    plugins: Sequence[OwnedPlugin] = (),
    generated_at: Optional[datetime] = None,
) -> Doc:
    """The whole report as blocks, in reading order."""
    _sanitise(analysis)

    when = generated_at or datetime.now(timezone.utc)
    genre_label = targets.get_profile(analysis.genre).label
    owned = capabilities.owned_capabilities(list(plugins))
    prescriptions = _prescriptions(analysis)

    findings = list(analysis.findings or [])
    # Anything the producer confirmed as deliberate leaves the fix pipeline
    # entirely — it is not in *Start here*, not in the deviations, not on the
    # session plan, and not in the concepts. It gets its own short section as a
    # decision. A document that lists something under "what to change" after
    # the producer has said they meant it is the exact failure the question was
    # added to prevent.
    confirmed = sorted([f for f in findings if _confirmed(f)], key=_rank)
    live = [f for f in findings if not _confirmed(f)]
    defects = sorted([f for f in live if _kind_of(f) == "defect"], key=_rank)
    deviations = sorted([f for f in live if _kind_of(f) != "defect"], key=_rank)
    ordered = defects + deviations

    doc = Doc()
    # The explainers define these themselves, so the document's own prose must
    # not define them a second time.
    for word in ("true peak", "dBTP", "LUFS", "inter-sample peak"):
        doc.seen(word)

    num = _Numbering()
    _cover(doc, analysis, defects, deviations, confirmed, genre_label, when)
    _start_here(doc, analysis, ordered, owned, prescriptions, genre_label, num)
    _confirmed_section(doc, confirmed, num)
    _defect_section(doc, defects, owned, prescriptions, num)
    _deviation_section(doc, deviations, owned, prescriptions, analysis, genre_label,
                       num, confirmed)
    _working_section(doc, analysis, genre_label, num)
    _session_plan(doc, ordered, owned, prescriptions, analysis, num)
    _concepts_section(doc, ordered, genre_label, num)
    _appendix(doc, analysis, genre_label, num)

    doc.add(Rule())
    doc.p(
        f"Generated by Mix Diagnostic on {when.strftime('%d %B %Y')} from "
        f"{analysis.filename or 'an uploaded file'}, measured against the {genre_label} "
        f"reference as a {TRACK_INTENT_LABELS.get(analysis.intent, analysis.intent).lower()}. "
        "Every number in this document was measured from the audio; nothing was estimated or "
        "invented. Re-run the analysis on your next bounce to see what moved."
    )
    return doc


def render_markdown(
    analysis: MixAnalysis,
    plugins: Sequence[OwnedPlugin] = (),
    generated_at: Optional[datetime] = None,
) -> str:
    return _to_markdown(build_document(analysis, plugins, generated_at))


def render_html(
    analysis: MixAnalysis,
    plugins: Sequence[OwnedPlugin] = (),
    generated_at: Optional[datetime] = None,
) -> str:
    doc = build_document(analysis, plugins, generated_at)
    stem = (analysis.filename or "mix").rsplit(".", 1)[0]
    return _to_html(doc, f"{stem} — Mix Diagnostic report")


def suggested_filename(analysis: MixAnalysis, extension: str = "md") -> str:
    """`{track-name}-mixdoctor-report.md`, safe on every filesystem."""
    stem = (analysis.filename or "mix").rsplit(".", 1)[0]
    slug = _slug(stem) or "mix"
    return f"{slug[:80]}-mixdoctor-report.{extension}"
