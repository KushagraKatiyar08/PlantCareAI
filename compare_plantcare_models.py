from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_DIR = Path(__file__).resolve().parent
BASELINE_CSV = PROJECT_DIR / "evaluation_results_baseline" / "per_image_predictions.csv"
CANDIDATE_CSV = PROJECT_DIR / "evaluation_results_finetuned" / "per_image_predictions.csv"
OUTPUT_DIR = PROJECT_DIR / "model_comparison"


def read_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required results file is missing: {path}\n"
            "Run the baseline and fine-tuned evaluation scripts first."
        )
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {
            "category_folder", "image", "actual_label", "predicted_label",
            "top1_score_percent", "top3_correct",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        rows = {}
        for row in reader:
            key = (row["category_folder"], row["image"])
            if key in rows:
                raise ValueError(f"Duplicate image entry in {path.name}: {key}")
            rows[key] = row
        return rows


def calculate_metrics(rows: list[dict[str, str]], target_labels: list[str]) -> dict[str, float]:
    y_true = [row["actual_label"] for row in rows]
    y_pred = [row["predicted_label"] for row in rows]
    return {
        "images": float(len(rows)),
        "top1_accuracy": accuracy_score(y_true, y_pred),
        "top3_accuracy": sum(
            row["top3_correct"].strip().lower() in {"true", "1", "yes"}
            for row in rows
        ) / max(len(rows), 1),
        "macro_precision": precision_score(
            y_true, y_pred, labels=target_labels, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, labels=target_labels, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, labels=target_labels, average="macro", zero_division=0
        ),
    }


def main() -> None:
    baseline = read_rows(BASELINE_CSV)
    candidate = read_rows(CANDIDATE_CSV)

    baseline_keys = set(baseline)
    candidate_keys = set(candidate)
    if baseline_keys != candidate_keys:
        missing_candidate = sorted(baseline_keys - candidate_keys)
        missing_baseline = sorted(candidate_keys - baseline_keys)
        print("WARNING: The image sets do not match exactly.")
        print(f"Images missing from candidate results: {len(missing_candidate)}")
        print(f"Images missing from baseline results: {len(missing_baseline)}")
        if missing_candidate:
            print("Example missing candidate image:", missing_candidate[0])
        if missing_baseline:
            print("Example missing baseline image:", missing_baseline[0])

    shared_keys = sorted(baseline_keys & candidate_keys)
    if not shared_keys:
        raise RuntimeError("No shared images found; refusing to compare unrelated test sets.")

    for key in shared_keys:
        if baseline[key]["actual_label"] != candidate[key]["actual_label"]:
            raise ValueError(f"Ground-truth label mismatch for image: {key}")

    baseline_rows = [baseline[key] for key in shared_keys]
    candidate_rows = [candidate[key] for key in shared_keys]
    target_labels = sorted({row["actual_label"] for row in baseline_rows})
    old_metrics = calculate_metrics(baseline_rows, target_labels)
    new_metrics = calculate_metrics(candidate_rows, target_labels)

    metric_labels = {
        "top1_accuracy": "Top-1 accuracy",
        "top3_accuracy": "Top-3 accuracy",
        "macro_precision": "Macro precision",
        "macro_recall": "Macro recall",
        "macro_f1": "Macro F1-score",
    }

    output_lines = [
        "PLANTCARE AI — BASELINE VS FINE-TUNED MODEL",
        "=" * 60,
        f"Images evaluated in both runs: {len(shared_keys)}",
        f"Baseline result rows: {len(baseline)}",
        f"Candidate result rows: {len(candidate)}",
        "",
        f"{'Metric':<24} {'Baseline':>12} {'Candidate':>12} {'Change (pp)':>13}",
        "-" * 64,
    ]
    for key, label in metric_labels.items():
        old = old_metrics[key]
        new = new_metrics[key]
        change_pp = (new - old) * 100
        output_lines.append(
            f"{label:<24} {old * 100:>11.2f}% {new * 100:>11.2f}% {change_pp:>+12.2f}"
        )

    class_data: list[dict[str, object]] = []
    baseline_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    candidate_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in baseline_rows:
        baseline_by_class[row["category_folder"]].append(row)
    for row in candidate_rows:
        candidate_by_class[row["category_folder"]].append(row)

    output_lines += ["", "PER-CATEGORY TOP-1 RESULTS", "-" * 64]
    output_lines.append(
        f"{'Category':<31} {'Base':>8} {'Candidate':>10} {'Change (pp)':>12} {'N':>5}"
    )

    for category in sorted(baseline_by_class):
        old_rows = baseline_by_class[category]
        new_rows = candidate_by_class[category]
        old_acc = sum(r["actual_label"] == r["predicted_label"] for r in old_rows) / len(old_rows)
        new_acc = sum(r["actual_label"] == r["predicted_label"] for r in new_rows) / len(new_rows)
        delta = (new_acc - old_acc) * 100
        class_data.append({
            "category_folder": category,
            "images": len(old_rows),
            "baseline_correct": sum(r["actual_label"] == r["predicted_label"] for r in old_rows),
            "candidate_correct": sum(r["actual_label"] == r["predicted_label"] for r in new_rows),
            "baseline_top1_accuracy_percent": round(old_acc * 100, 2),
            "candidate_top1_accuracy_percent": round(new_acc * 100, 2),
            "change_percentage_points": round(delta, 2),
        })
        display_category = category[:30]
        output_lines.append(
            f"{display_category:<31} {old_acc * 100:>7.2f}% {new_acc * 100:>9.2f}% {delta:>+11.2f} {len(old_rows):>5}"
        )

    improved = sum(float(row["change_percentage_points"]) > 0 for row in class_data)
    unchanged = sum(float(row["change_percentage_points"]) == 0 for row in class_data)
    declined = sum(float(row["change_percentage_points"]) < 0 for row in class_data)
    old_correct = sum(row["actual_label"] == row["predicted_label"] for row in baseline_rows)
    new_correct = sum(row["actual_label"] == row["predicted_label"] for row in candidate_rows)
    delta_correct = new_correct - old_correct

    output_lines += [
        "",
        "INTERPRETATION CHECK",
        "-" * 64,
        f"Categories improved: {improved}",
        f"Categories unchanged: {unchanged}",
        f"Categories declined: {declined}",
        f"Net change in correct Top-1 predictions: {delta_correct:+d} images",
        "",
        "Use this comparison as an experiment result, not proof of general field accuracy.",
        "The benchmark has only the selected mapped categories and 105 images at most.",
        "Verify the category labels before claiming disease-specific diagnostic performance.",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "comparison_summary.txt").write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    with (OUTPUT_DIR / "per_category_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=[
            "category_folder", "images", "baseline_correct", "candidate_correct",
            "baseline_top1_accuracy_percent", "candidate_top1_accuracy_percent",
            "change_percentage_points",
        ])
        writer.writeheader()
        writer.writerows(class_data)

    print("\n".join(output_lines))
    print(f"\nFiles saved in: {OUTPUT_DIR}")
    print(" - comparison_summary.txt")
    print(" - per_category_comparison.csv")


if __name__ == "__main__":
    main()
