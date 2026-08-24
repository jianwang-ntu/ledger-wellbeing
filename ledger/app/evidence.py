"""What this project has actually established about each dimension.

The application shows numbers. Numbers are persuasive in a way a JSON file in a
repository is not, so every dimension the product displays has to arrive with
the strength of the evidence behind it attached — pre-registered as **R8-7**.

Nothing here is written by hand. The per-dimension figures are read from
`artifacts/scorer_ablation.json`, which is the tracked output of the increment-4
measurement, so the application cannot drift into claiming more than was
measured. If the artifact is missing, the application reports the evidence as
unavailable rather than defaulting to something flattering.

The threshold is `USABLE_HELD_OUT_AUC = 0.70`, fixed in build increment 3
*before* any of these numbers existed. `activation` sits at 0.600 and is
therefore **not** a dimension this project has shown to work, whatever the macro
average is.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ledger.model.dimensions import DIMENSIONS

ROOT = Path(__file__).resolve().parents[2]
SCORER_ABLATION = ROOT / "artifacts" / "scorer_ablation.json"

#: Fixed in build increment 3, before the increment-4 numbers existed.
USABLE_HELD_OUT_AUC = 0.70

#: What the held-out AUC was measured on. Quoted wherever a number is shown.
EVALUATION_BASIS = (
    "25 withheld anchor-sentence pairs per dimension, from anchors written for "
    "this repository. Not clinical data, not real journal entries, and not an "
    "external benchmark."
)

NOT_ESTABLISHED_NOTE = (
    "Below the 0.70 held-out threshold fixed before this was measured. This "
    "project has not shown this dimension works; it is shown only so that its "
    "weakness is visible rather than averaged away."
)


@lru_cache(maxsize=1)
def dimension_evidence() -> dict:
    """Per-dimension held-out AUC and whether it clears the pre-fixed floor."""
    if not SCORER_ABLATION.exists():
        return {
            dim: {"held_out_auc": None, "established": False,
                  "note": "artifacts/scorer_ablation.json is absent; no evidence available"}
            for dim in DIMENSIONS
        }

    report = json.loads(SCORER_ABLATION.read_text())
    selected = report.get("selected_model")
    per_dimension = {}
    for variant in report.get("variants", []):
        if variant.get("model") == selected and variant.get("per_dimension"):
            per_dimension = variant["per_dimension"]
            break

    out = {}
    for dim in DIMENSIONS:
        auc = per_dimension.get(dim, {}).get("held_out_auc")
        established = auc is not None and auc >= USABLE_HELD_OUT_AUC
        out[dim] = {
            "held_out_auc": auc,
            "established": established,
            "threshold": USABLE_HELD_OUT_AUC,
            "basis": EVALUATION_BASIS,
            "note": None if established else NOT_ESTABLISHED_NOTE,
        }
    return out


def established_dimensions() -> tuple[str, ...]:
    return tuple(d for d, e in dimension_evidence().items() if e["established"])


def unestablished_dimensions() -> tuple[str, ...]:
    return tuple(d for d, e in dimension_evidence().items() if not e["established"])
