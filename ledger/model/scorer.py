"""LedgerScorer — a small encoder with an *exactly additive* attribution head.

The point of this module is criterion C4 of plan.md: "explainable AI mechanisms
to show why the model reached a health conclusion". Most saliency methods are
approximations of the model they explain — you get a heatmap and you take it on
trust. This head is built so that the explanation is not an approximation of the
score, it *is* the score:

    logit_k = sum over tokens i of ( mask_i * (h_i . w_k) / N ) + b_k
            = sum over tokens i of ( token_attr[i, k] )          + b_k

Mean pooling followed by a linear head is a sum of per-token terms, so the
per-token contributions add up to the logit exactly, to floating-point error.
`tests/test_attribution_identity.py` asserts that identity on every build,
including the int8 one, so a regression in the export pipeline cannot quietly
turn the explanation into a decoration.

The head itself is *zero-shot* in this build: each row is the difference of two
anchor-phrase centroids from `dimensions.ANCHORS`, affinely calibrated so the
negative pole sits near logit -2 and the positive pole near +2. That is a
prototype classifier, not a trained one, and it is labelled `anchor_v0` so that
no artifact produced from it can be mistaken for a fine-tune. Supervised
fine-tuning is a later build increment and is gated on plan.md R-1 (corpus
licence).
"""

from __future__ import annotations

import torch
from torch import nn

from .dimensions import ANCHORS, DIMENSIONS, HEAD_VERSION

#: Where the negative and positive anchor poles are placed in logit space.
NEG_POLE = -2.0
POS_POLE = 2.0


class LedgerScorer(nn.Module):
    """Encoder + mean pooling + linear head, emitting logits and token attributions."""

    def __init__(self, encoder: nn.Module, hidden_size: int, num_dimensions: int):
        super().__init__()
        self.encoder = encoder
        self.weight = nn.Parameter(torch.zeros(num_dimensions, hidden_size))
        self.bias = nn.Parameter(torch.zeros(num_dimensions))
        self.head_version = HEAD_VERSION

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)          # B,T,1
        n_tokens = mask.sum(dim=1).clamp(min=1.0)                      # B,1
        contrib = torch.matmul(hidden, self.weight.t())                # B,T,K
        token_attr = contrib * mask / n_tokens.unsqueeze(1)            # B,T,K
        logits = token_attr.sum(dim=1) + self.bias                     # B,K
        return logits, token_attr


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


@torch.no_grad()
def build_anchor_head(encoder: nn.Module, tokenizer, device: str = "cpu", max_length: int = 64):
    """Derive (weight, bias) from the anchor phrases. Deterministic — no training.

    Returns the head tensors plus a per-dimension record of where the two poles
    actually landed, so the calibration can be audited rather than assumed.
    """
    encoder = encoder.to(device).eval()
    weight_rows, bias_values, report = [], [], {}

    for dim in DIMENSIONS:
        poles = {}
        for pole in ("positive", "negative"):
            batch = tokenizer(
                ANCHORS[dim][pole],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            hidden = encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])[0]
            poles[pole] = mean_pool(hidden, batch["attention_mask"])    # P,H

        direction = poles["positive"].mean(0) - poles["negative"].mean(0)  # H
        proj_pos = float(poles["positive"].mean(0) @ direction)
        proj_neg = float(poles["negative"].mean(0) @ direction)
        spread = proj_pos - proj_neg
        if spread <= 0:
            raise ValueError(f"anchor poles for {dim!r} did not separate (spread={spread})")

        scale = (POS_POLE - NEG_POLE) / spread
        offset = NEG_POLE - scale * proj_neg

        weight_rows.append((direction * scale).cpu())
        bias_values.append(offset)
        report[dim] = {
            "raw_projection_positive_pole": proj_pos,
            "raw_projection_negative_pole": proj_neg,
            "raw_spread": spread,
            "scale": scale,
            "offset": offset,
            "n_anchor_phrases": len(ANCHORS[dim]["positive"]) + len(ANCHORS[dim]["negative"]),
        }

    return torch.stack(weight_rows), torch.tensor(bias_values, dtype=torch.float32), report
