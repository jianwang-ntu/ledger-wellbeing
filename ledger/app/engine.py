"""The end-to-end path: text in, an explained and stored observation out.

    text → deterministic crisis route → (score → per-span attribution) → record

Two ordering properties are load-bearing and are measured rather than asserted:

**The router runs first, and on an acute match the model does not run at all.**
Pre-registered as **R8-4** and counted in `tests/test_engine.py` by instrumenting
the ONNX session itself. `plan.md`'s third differentiator is "a deterministic
safety layer that is not the model"; a safety layer that runs *alongside* the
model is a different and weaker thing than one that runs *instead of* it.

**Nothing here reaches the network.** `ledger.app.offline` is imported first, for
its side effects, and `export/egress_audit.py` measures the result on the running
application.

What this module does not do, and will not be extended to do: generate text,
give advice, name a condition, or suggest a treatment. It reports what the
person wrote and which of their own words moved a score.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

from ledger.app import offline  # noqa: F401  - must precede the transformers import

import numpy as np
import onnxruntime as ort

from ledger.app.evidence import dimension_evidence
from ledger.app.spans import attribute_spans, sentence_spans, word_spans
from ledger.model.dimensions import DIMENSION_LABELS, DIMENSIONS
from ledger.safety.crisis_router import route
from ledger.store.journal import JournalEntry, utc_now_iso

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
TOKENIZER_DIR = ARTIFACTS / "tokenizer"

#: The build `export/verify.py` selected. Changing this changes what ships, so
#: it is read from the verify report rather than hard-coded twice.
VERIFY_REPORT = ARTIFACTS / "verify_report.json"
BUILD_FILES = {
    "int8_embed": ARTIFACTS / "onnx" / "ledger_scorer_int8embed.onnx",
    "int8_full": ARTIFACTS / "onnx" / "ledger_scorer_int8.onnx",
    "fp32": ARTIFACTS / "onnx" / "ledger_scorer_fp32_single.onnx",
}

#: CEIL-4 is stated at this length, so inference is done at this length. Padding
#: tokens are masked to exactly zero contribution upstream, so they cost time and
#: change no number.
MAX_LENGTH = 256

#: R8-2. The same tolerance as R6-1; the aggregation step does not get its own.
ADDITIVITY_MAX_RESIDUAL = 1e-4

NON_DIAGNOSTIC_CONTRACT = (
    "Ledger is not a medical device and does not diagnose. It reports patterns in "
    "language you wrote, so that you can decide what to share with a clinician who "
    "is treating you. It does not replace professional care."
)


class ModelUnavailable(RuntimeError):
    """The selected build is not on disk. Raised instead of scoring silently badly."""


@dataclass
class DimensionResult:
    dimension: str
    label: str
    logit: float
    probability: float
    established: bool
    held_out_auc: float | None
    evidence_note: str | None
    spans: list = field(default_factory=list)
    structural_attribution: float = 0.0
    structural_tokens: int = 0
    additivity_residual: float = 0.0


@dataclass
class Analysis:
    entry_id: str
    written_at: str
    text: str
    routed: dict
    scored: bool
    reason_not_scored: str | None
    granularity: str
    dimensions: list = field(default_factory=list)
    model: dict = field(default_factory=dict)
    contract: str = NON_DIAGNOSTIC_CONTRACT

    def to_record(self) -> JournalEntry:
        payload = asdict(self)
        for own in ("entry_id", "written_at", "text"):
            payload.pop(own)
        return JournalEntry(entry_id=self.entry_id, written_at=self.written_at,
                            text=self.text, analysis=payload)


def selected_build() -> str:
    if VERIFY_REPORT.exists():
        report = json.loads(VERIFY_REPORT.read_text())
        name = report.get("selected_build")
        if name in BUILD_FILES:
            return name
    return "int8_embed"


class LedgerEngine:
    """Loads once, analyses many. Holds no user text between calls."""

    def __init__(self, build: str | None = None, region: str | None = None):
        self.build = build or selected_build()
        self.region = region
        self.model_path = BUILD_FILES[self.build]
        self._session: ort.InferenceSession | None = None
        self._tokenizer = None
        self._bias: np.ndarray | None = None

    # -- lazy loading ------------------------------------------------------
    # Loading is deferred so that the crisis path, the store and the report can
    # all be exercised without paying 200 MB of model load — and so that R8-4 is
    # observable: on an acute entry the session is never even constructed.

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_DIR, local_files_only=True
            )
        return self._tokenizer

    @property
    def session(self) -> ort.InferenceSession:
        if self._session is None:
            if not self.model_path.exists():
                raise ModelUnavailable(
                    f"{self.model_path} is absent. Regenerate it with "
                    "`bash export/run_all.sh`; it is a build output and is not tracked."
                )
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(self.model_path), opts, providers=["CPUExecutionProvider"]
            )
        return self._session

    @property
    def bias(self) -> np.ndarray:
        if self._bias is None:
            import torch
            ckpt = torch.load(ARTIFACTS / "torch" / "head.pt",
                              map_location="cpu", weights_only=True)
            self._bias = ckpt["head_bias"].numpy().astype(np.float64)
        return self._bias

    def model_card(self) -> dict:
        return {
            "build": self.build,
            "head_version": "anchor_v0",
            "head_is_trained": False,
            "head_note": (
                "The head is a zero-shot anchor head, not a fine-tune. plan.md R-1 "
                "(no permissively-licensed corpus cleared) blocks training and this "
                "build claims none."
            ),
            "runs_locally": True,
            "sends_anything_anywhere": False,
        }

    # -- the path ----------------------------------------------------------

    def analyse(self, text: str, *, entry_id: str | None = None,
                written_at: str | None = None, region: str | None = None,
                granularity: str = "sentence") -> Analysis:
        entry_id = entry_id or uuid4().hex[:16]
        written_at = written_at or utc_now_iso()

        # 1. The router. Before tokenization, before any model exists.
        decision = route(text, region if region is not None else self.region)
        routed = {
            "triggered": decision.triggered,
            "severity": decision.severity,
            "rule_ids": list(decision.rule_ids),
            "blocks_model_output": decision.blocks_model_output,
            "helplines": [
                {"region": h.region, "name": h.name, "contact": h.contact, "hours": h.hours}
                for h in decision.helplines
            ],
        }

        if decision.blocks_model_output:
            # R8-4. Not "the score is hidden" — the score is never computed. The
            # instrument's job here is to put a person in front of a person.
            return Analysis(
                entry_id=entry_id, written_at=written_at, text=text, routed=routed,
                scored=False, granularity=granularity,
                reason_not_scored=(
                    "An acute crisis rule matched. Producing a trend line about "
                    "someone in that moment is not what this tool is for."
                ),
                model=self.model_card(),
            )

        # 2. Score and attribute.
        dimensions = self._score(text, granularity)
        return Analysis(
            entry_id=entry_id, written_at=written_at, text=text, routed=routed,
            scored=True, reason_not_scored=None, granularity=granularity,
            dimensions=[asdict(d) for d in dimensions], model=self.model_card(),
        )

    def _score(self, text: str, granularity: str) -> list[DimensionResult]:
        spans = sentence_spans(text) if granularity == "sentence" else word_spans(text)

        encoded = self.tokenizer(
            [text], padding="max_length", truncation=True, max_length=MAX_LENGTH,
            return_offsets_mapping=True, return_tensors="np",
        )
        feeds = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        logits, token_attr = self.session.run(["logits", "token_attr"], feeds)
        logits = logits.astype(np.float64)[0]
        token_attr = token_attr.astype(np.float64)[0]          # T,K
        offsets = encoded["offset_mapping"][0]
        mask = encoded["attention_mask"][0]

        evidence = dimension_evidence()
        results = []
        for k, dim in enumerate(DIMENSIONS):
            span_attr, structural, n_structural = attribute_spans(
                text, offsets, mask, token_attr[:, k], spans
            )
            total = sum(s.attribution for s in span_attr) + structural + float(self.bias[k])
            residual = abs(float(logits[k]) - total)
            results.append(DimensionResult(
                dimension=dim,
                label=DIMENSION_LABELS[dim],
                logit=float(logits[k]),
                probability=float(1.0 / (1.0 + np.exp(-logits[k]))),
                established=evidence[dim]["established"],
                held_out_auc=evidence[dim]["held_out_auc"],
                evidence_note=evidence[dim]["note"],
                spans=[asdict(s) for s in span_attr],
                structural_attribution=structural,
                structural_tokens=n_structural,
                additivity_residual=residual,
            ))
        return results
