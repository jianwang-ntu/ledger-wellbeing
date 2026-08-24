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
application. Since DEFECT-INC8-001 the application also carries no library with a
hub-checking code path at all: tokenization goes through `tokenizers` directly and
the head bias is read from a build report, so `transformers` and `torch` are build
dependencies only.

What this module does not do, and will not be extended to do: generate text,
give advice, name a condition, or suggest a treatment. It reports what the
person wrote and which of their own words moved a score.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

from ledger.app import offline  # noqa: F401  - side effect: pin the ML libraries offline

import numpy as np
import onnxruntime as ort

from ledger.app.evidence import dimension_evidence
from ledger.app import local_tokenizer
from ledger.app.local_tokenizer import MAX_LENGTH, encode as encode_text
from ledger.app.spans import attribute_spans, sentence_spans, word_spans
from ledger.model.dimensions import DIMENSION_LABELS, DIMENSIONS
from ledger.safety.crisis_router import route
from ledger.store.journal import JournalEntry, utc_now_iso

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
BUILD_REPORT = ARTIFACTS / "torch" / "build_report.json"

#: The build `export/verify.py` selected. Changing this changes what ships, so
#: it is read from the verify report rather than hard-coded twice.
VERIFY_REPORT = ARTIFACTS / "verify_report.json"
BUILD_FILES = {
    "int8_embed": ARTIFACTS / "onnx" / "ledger_scorer_int8embed.onnx",
    "int8_full": ARTIFACTS / "onnx" / "ledger_scorer_int8.onnx",
    "fp32": ARTIFACTS / "onnx" / "ledger_scorer_fp32_single.onnx",
}

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
    #: The per-dimension offset from anchor calibration. Carried explicitly so a
    #: reader can add the displayed terms up and land on the score, rather than
    #: being told they add up.
    bias: float = 0.0
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
        """The `tokenizers` fast tokenizer, loaded from the local build output.

        Not `transformers`: see `ledger/app/local_tokenizer.py`, DEFECT-INC8-001.
        Encoding parity with the path every prior measurement used is measured by
        `export/tokenizer_parity.py`, not assumed.
        """
        if self._tokenizer is None:
            self._tokenizer = local_tokenizer.load(MAX_LENGTH)
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
        """The head bias, read from the build report rather than from `head.pt`.

        `head.pt` stores a float32 cast of these same calibration offsets, so the
        cast below reproduces the shipped bias bit-for-bit and the application
        does not import `torch` to obtain five numbers.
        `export/tokenizer_parity.py` measures that equality.
        """
        if self._bias is None:
            if not BUILD_REPORT.exists():
                raise ModelUnavailable(
                    f"{BUILD_REPORT} is absent; it is a build output. "
                    "Regenerate it with `bash export/run_all.sh`."
                )
            report = json.loads(BUILD_REPORT.read_text())
            calibration = report["anchor_calibration"]
            self._bias = np.array(
                [calibration[dim]["offset"] for dim in DIMENSIONS], np.float32
            ).astype(np.float64)
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

        input_ids, mask, offsets = encode_text(text, MAX_LENGTH)
        feeds = {
            "input_ids": np.asarray([input_ids], dtype=np.int64),
            "attention_mask": np.asarray([mask], dtype=np.int64),
        }
        logits, token_attr = self.session.run(["logits", "token_attr"], feeds)
        logits = logits.astype(np.float64)[0]
        token_attr = token_attr.astype(np.float64)[0]          # T,K

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
                bias=float(self.bias[k]),
                additivity_residual=residual,
            ))
        return results
