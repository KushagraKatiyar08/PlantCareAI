"""Prediction history persistence and statistics utilities for PlantCare AI."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

HISTORY_FILE = Path(__file__).parent / "prediction_history.csv"

FIELDNAMES = [
    "timestamp",
    "image_filename",
    "predicted_category",
    "confidence_percent",
]


def save_prediction(image_filename: str, label: str, confidence: float, user_id: int | None = None, **kwargs: Any) -> None:
    """Save an analysis record to the CSV history file and SQLite database preserving backward compatibility."""
    file_exists = HISTORY_FILE.exists() and HISTORY_FILE.stat().st_size > 0
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Preserve existing CSV logging
    with HISTORY_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": now_str,
            "image_filename": image_filename,
            "predicted_category": label,
            "confidence_percent": round(confidence * 100, 2),
        })

    # 2. Additive database record ownership
    try:
        from auth_db import init_db, save_prediction_record
        init_db()
        target_uid = user_id if user_id is not None else 1
        save_prediction_record(
            user_id=target_uid,
            image_filename=image_filename,
            predicted_category=label,
            confidence_percent=confidence * 100 if confidence <= 1.0 else confidence,
        )
        try:
            import streamlit as st
            st.cache_data.clear()
        except Exception:
            pass
    except Exception:
        pass


def load_history(user_id: int | None = None, is_admin: bool = False) -> list[dict[str, Any]]:
    """Load prediction records. If user_id is specified, loads private user records from database."""
    if user_id is not None:
        try:
            from auth_db import get_predictions_for_user, init_db
            init_db()
            return get_predictions_for_user(user_id=user_id, is_admin=is_admin)
        except Exception:
            pass

    if not HISTORY_FILE.exists():
        return []

    try:
        with HISTORY_FILE.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return list(reader)
    except (OSError, csv.Error):
        return []


def get_history_stats(user_id: int | None = None, is_admin: bool = False) -> dict[str, Any]:
    """Calculate aggregate statistics from real prediction history respecting ownership."""
    if user_id is not None:
        try:
            from auth_db import get_stats_for_user, init_db
            init_db()
            return get_stats_for_user(user_id=user_id, is_admin=is_admin)
        except Exception:
            pass

    records = load_history()
    if not records:
        return {
            "total_analyses": 0,
            "distinct_classes": 0,
            "latest_date": "No records yet",
            "avg_confidence": 0.0,
        }

    total = len(records)
    categories = {r.get("predicted_category", "") for r in records if r.get("predicted_category")}

    confidences = []
    for r in records:
        try:
            confidences.append(float(r.get("confidence_percent", 0)))
        except (ValueError, TypeError):
            pass

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    latest = records[-1].get("timestamp", "Recent")

    return {
        "total_analyses": total,
        "distinct_classes": len(categories),
        "latest_date": latest,
        "avg_confidence": round(avg_conf, 1),
    }


def filter_history(records: list[dict[str, Any]], query: str = "") -> list[dict[str, Any]]:
    """Filter prediction history by category or filename."""
    if not query:
        return records

    q = query.strip().lower()
    return [
        r for r in records
        if q in str(r.get("predicted_category", "")).lower()
        or q in str(r.get("image_filename", "")).lower()
        or q in str(r.get("timestamp", "")).lower()
    ]
