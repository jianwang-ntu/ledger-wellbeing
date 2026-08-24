"""When may `export/common.py:BASE_MODEL` move off the incumbent?

Increments 3, 4 and 5 each wrote a guard of the form *"a null or blocked
selection leaves common.py on the incumbent"*. Increment 6 moved the pin, and all
three of those guards failed — correctly. They were doing their job.

The guards are not deleted, because "the pin never moves" was never the real
invariant; it was a cheap approximation of one while the delivery target was
fixed. The real invariant, which is what those guards were protecting, is:

    A pin only moves to a model that an ablation actually SELECTED, and only
    once every adoption blocker that ablation recorded is a ceiling that is no
    longer ENFORCED on the current delivery target.

That is strictly stronger than "never move" in the direction that matters, and it
cannot be satisfied by fiat:

* It fails if the pin moves to something no ablation chose — so the increment-3
  and increment-5 null selections still cannot move it. Both of those ablations
  selected `null`, so nothing they measured is ever a legal destination.
* It fails if a *currently enforced* ceiling is among the recorded blockers — so a
  scorer blocked on CEIL-2, CEIL-4 or CEIL-5 cannot be adopted on the desktop
  target no matter how the target moves.
* It reads `ENFORCED_BY_TARGET`, which `tests/test_delivery_target.py` pins to
  exactly `{CEIL-1, CEIL-3}` dropped and no more.

Increment 6's move is legal under it: `scorer_ablation.json` selected
`nli_sbert_centroid` and recorded exactly two blockers, `CEIL_1_int8_model_bytes`
and `CEIL_3_cold_payload_bytes`, both of which the desktop target stops enforcing.
It is worth being explicit that this makes the pin legal, **not** the build
shippable — increment 6 adopted no build, because CEIL-2 fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "export"))

ARTIFACTS = ROOT / "artifacts"


def _selections() -> list[tuple[str, str, list[str]]]:
    """(model, revision, recorded_blockers) for every ablation that selected one."""
    out = []

    path = ARTIFACTS / "scorer_ablation.json"
    if path.exists():
        rep = json.loads(path.read_text())
        if rep.get("selected"):
            chosen = next(v for v in rep["variants"] if v["variant"] == rep["selected"])
            out.append((chosen["model"], chosen["revision"],
                        list(rep.get("adoption", {}).get("adoption_blockers", []))))

    for name in ("encoder_ablation.json", "size_feasible_scorer.json"):
        path = ARTIFACTS / name
        if not path.exists():
            continue
        rep = json.loads(path.read_text())
        if rep.get("selected"):
            out.append((rep["selected"], rep.get("selected_revision"), []))
    return out


def pin_is_legal() -> tuple[bool, str]:
    """Is the current BASE_MODEL pin permitted? Returns (verdict, reason)."""
    from common import BASE_MODEL, BASE_REVISION, enforced_ceilings

    enforced = set(enforced_ceilings())
    for model, revision, blockers in _selections():
        if model != BASE_MODEL:
            continue
        if revision and revision != BASE_REVISION:
            return False, (f"pin {BASE_MODEL} matches a selection but at revision "
                           f"{BASE_REVISION}, not the selected {revision}")
        still_binding = sorted(set(blockers) & enforced)
        if still_binding:
            return False, (f"pin moved to {model}, whose recorded adoption blockers "
                           f"{still_binding} are STILL enforced on this delivery target")
        return True, (f"pin {model} was selected by an ablation; its recorded blockers "
                      f"{sorted(blockers)} are not enforced on this delivery target")
    return False, (f"pin {BASE_MODEL} was not selected by any ablation on record; "
                   "a null selection may not move the pin")


def pin_is_incumbent(incumbent_model: str, incumbent_revision: str) -> bool:
    from common import BASE_MODEL, BASE_REVISION
    return BASE_MODEL == incumbent_model and BASE_REVISION == incumbent_revision
