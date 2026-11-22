"""Conservative review flags for image-classification predictions.

The thresholds are heuristic defaults, not statistically calibrated diagnostic
cut-offs. Validate them on a separate, representative validation dataset before
using them for any real-world decision.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

DEFAULT_MIN_TOP_SCORE = 0.60
DEFAULT_MIN_SCORE_MARGIN = 0.12


def assess_prediction(
    predictions: Iterable[dict[str, Any]],
    *,
    min_top_score: float = DEFAULT_MIN_TOP_SCORE,
    min_score_margin: float = DEFAULT_MIN_SCORE_MARGIN,
) -> dict[str, Any]:
    """Assess a Hugging Face image-classification top-k result.

    Returns a dictionary with label, score, score margin, review flag, heading,
    and user-facing explanation. The function does not claim that a score is a
    calibrated probability or that an image is definitely a plant.
    """
    if not 0 <= min_top_score <= 1:
        raise ValueError("min_top_score must be between 0 and 1")
    if not 0 <= min_score_margin <= 1:
        raise ValueError("min_score_margin must be between 0 and 1")

    clean: list[tuple[str, float]] = []
    for item in predictions:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        if label and math.isfinite(score) and 0 <= score <= 1:
            clean.append((label, score))

    clean.sort(key=lambda entry: entry[1], reverse=True)
    if not clean:
        return {
            "label": None,
            "score": None,
            "score_percent": None,
            "margin": None,
            "margin_percent": None,
            "needs_review": True,
            "heading": "No usable prediction",
            "message": (
                "The model did not return a usable result. Try a clear, well-lit "
                "photo of one leaf, or select another image."
            ),
            "top_predictions": [],
        }

    top_label, top_score = clean[0]
    second_score = clean[1][1] if len(clean) > 1 else 0.0
    margin = max(0.0, top_score - second_score)
    needs_review = top_score < min_top_score or margin < min_score_margin

    if needs_review:
        heading = "Uncertain result — review needed"
        message = (
            "The model is not sufficiently decisive under the current review "
            "rules. Do not treat this result as a confirmed disease diagnosis. "
            "Retake a clear photo of the affected leaf and compare the symptoms "
            "with a trusted agricultural source or local agriculture expert."
        )
    else:
        heading = "Candidate prediction — not a confirmed diagnosis"
        message = (
            "The model's top score and gap to its second choice passed the "
            "current heuristic review rules. This still does not prove the "
            "prediction is correct, and the model may fail on unsupported plants, "
            "healthy leaves, unusual lighting, or unfamiliar backgrounds. "
            "Confirm symptoms before taking action."
        )

    return {
        "label": top_label,
        "score": top_score,
        "score_percent": round(top_score * 100, 2),
        "margin": margin,
        "margin_percent": round(margin * 100, 2),
        "needs_review": needs_review,
        "heading": heading,
        "message": message,
        "top_predictions": [
            {"label": label, "score": score, "score_percent": round(score * 100, 2)}
            for label, score in clean[:3]
        ],
    }
