"""The clinician-shareable report.

This is the product's actual output. Not advice, not a summary, not a generated
paragraph: a trajectory over the person's own entries, and the spans of their own
writing that moved each score, so that a professional who is treating them can
see between-visit experience instead of asking them to recall it.

Three rules this renderer enforces, so they cannot be lost in a template:

* **R8-7** — a dimension that has not cleared the 0.70 held-out floor is labelled
  as not established, every time it appears, with the number.
* **R8-8** — nothing here names a condition, a severity or a treatment. The
  banned-vocabulary test in `tests/test_report.py` runs over this rendered text,
  not over the source.
* The non-diagnostic contract appears in every report, at the top, not in a
  footnote.

The user decides what to share. The report is produced on demand into a file the
user names; nothing is uploaded, and there is nowhere for it to be uploaded to.
"""

from __future__ import annotations

from dataclasses import dataclass

from ledger.app.evidence import EVALUATION_BASIS, USABLE_HELD_OUT_AUC, dimension_evidence
from ledger.model.dimensions import DIMENSION_LABELS, DIMENSIONS

CONTRACT = (
    "This report is not a diagnosis and Ledger is not a medical device. It shows "
    "patterns in language its owner wrote, computed on their own machine. Nothing "
    "in it was sent anywhere. It is intended to be read alongside the person, not "
    "instead of them."
)

TOP_SPANS = 3


@dataclass(frozen=True)
class Point:
    written_at: str
    entry_id: str
    probability: float


def _series(entries: list[dict], dimension: str) -> list[Point]:
    out = []
    for entry in entries:
        analysis = entry.get("analysis", {})
        if not analysis.get("scored"):
            continue
        for dim in analysis.get("dimensions", []):
            if dim["dimension"] == dimension:
                out.append(Point(entry["written_at"], entry["entry_id"], dim["probability"]))
    return out


def _sparkline(values: list[float]) -> str:
    """A fixed-scale 0..1 bar. Fixed, because an autoscaled axis makes noise look
    like a trend, and this is exactly the kind of picture that gets over-read."""
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[min(int(v * len(blocks)), len(blocks) - 1)] for v in values)


def _top_spans(entries: list[dict], dimension: str, limit: int = TOP_SPANS) -> list[tuple]:
    spans = []
    for entry in entries:
        analysis = entry.get("analysis", {})
        if not analysis.get("scored"):
            continue
        for dim in analysis.get("dimensions", []):
            if dim["dimension"] != dimension:
                continue
            for span in dim.get("spans", []):
                spans.append((span["attribution"], span["text"].strip(), entry["written_at"]))
    spans.sort(key=lambda s: -s[0])
    return spans[:limit]


def render(entries: list[dict], *, region: str | None = None) -> str:
    """Render the whole report as plain text. Deterministic given the entries."""
    evidence = dimension_evidence()
    scored = [e for e in entries if e.get("analysis", {}).get("scored")]
    routed = [e for e in entries
              if e.get("analysis", {}).get("routed", {}).get("triggered")]

    lines: list[str] = []
    lines.append("LEDGER — between-visits language report")
    lines.append("=" * 62)
    lines.append("")
    for chunk in _wrap(CONTRACT, 62):
        lines.append(chunk)
    lines.append("")
    lines.append(f"Entries in this store : {len(entries)}")
    lines.append(f"Entries scored        : {len(scored)}")
    lines.append(f"Entries routed to help: {len(routed)}")
    if entries:
        lines.append(f"Covering              : {entries[0]['written_at']} .. "
                     f"{entries[-1]['written_at']}")
    lines.append("")

    if not scored:
        lines.append("No scored entries yet, so there is no trajectory to show.")
        lines.append("")
    else:
        lines.append("TRAJECTORY  (probability per entry, fixed 0..1 scale)")
        lines.append("-" * 62)
        for dim in DIMENSIONS:
            series = _series(entries, dim)
            if not series:
                continue
            mark = "" if evidence[dim]["established"] else "   [NOT ESTABLISHED]"
            auc = evidence[dim]["held_out_auc"]
            auc_text = "n/a" if auc is None else f"{auc:.2f}"
            lines.append(f"{DIMENSION_LABELS[dim]:<34}{mark}".rstrip())
            lines.append(f"  {_sparkline([p.probability for p in series])}"
                         f"   latest {series[-1].probability:.2f}"
                         f"   held-out AUC {auc_text}")
            if not evidence[dim]["established"]:
                for chunk in _wrap(
                    f"This dimension scores {auc_text} against the {USABLE_HELD_OUT_AUC:.2f} "
                    "threshold fixed before it was measured. It has not been shown to work "
                    "and should not be read as a signal.", 58
                ):
                    lines.append(f"    {chunk}")
            lines.append("")

        lines.append("WHAT MOVED EACH SCORE  (the writer's own words)")
        lines.append("-" * 62)
        for dim in DIMENSIONS:
            top = _top_spans(entries, dim)
            if not top:
                continue
            suffix = "" if evidence[dim]["established"] else "  [NOT ESTABLISHED]"
            lines.append(f"{DIMENSION_LABELS[dim]}{suffix}")
            for attribution, text, written_at in top:
                snippet = text if len(text) <= 52 else text[:49] + "..."
                lines.append(f"  {attribution:+.3f}  {written_at[:10]}  \"{snippet}\"")
            lines.append("")

    lines.append("HOW TO READ THIS")
    lines.append("-" * 62)
    for chunk in _wrap(
        "Each number is a contrast between two ways of writing, not a quantity of "
        "anything. The contributions listed above are the score: they sum to it "
        "exactly, so nothing shown here is a reconstruction of a decision made "
        "elsewhere.", 62
    ):
        lines.append(chunk)
    lines.append("")
    for chunk in _wrap(
        "Scores come from a zero-shot anchor head, not a trained model. It was "
        f"evaluated on {EVALUATION_BASIS}", 62
    ):
        lines.append(chunk)
    lines.append("")

    if routed:
        lines.append("SAFETY ROUTING")
        lines.append("-" * 62)
        one = len(routed) == 1
        lines.append(f"{len(routed)} entr{'y' if one else 'ies'} matched a crisis rule "
                     f"and {'was' if one else 'were'} routed to published helplines.")
        for chunk in _wrap(
            "Entries matching an acute rule are never scored. That is a rule in the "
            "code, not a model judgement, and it cannot be overridden by anything "
            "written in an entry.", 62
        ):
            lines.append(chunk)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _wrap(text: str, width: int) -> list[str]:
    words, line, out = text.split(), "", []
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out
