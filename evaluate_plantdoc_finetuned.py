from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import pipeline

PROJECT_DIR = Path(__file__).resolve().parent
TEST_DIR = PROJECT_DIR / "datasets" / "PlantDoc-test-only" / "test"
MODEL_DIR = PROJECT_DIR / "models" / "plantdoc_finetuned"
REPORT_DIR = PROJECT_DIR / "evaluation_results_finetuned"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CATEGORY_MAP = {
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "grape leaf black rot": "Grape___Black_rot",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
}


def main() -> None:
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"Held-out test directory not found: {TEST_DIR}")
    if not (MODEL_DIR / "config.json").is_file():
        raise FileNotFoundError(
            f"Fine-tuned candidate not found: {MODEL_DIR}\nRun python fine_tune_plantdoc.py first."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    device = 0 if torch.cuda.is_available() else -1
    print(f"Held-out test folder: {TEST_DIR}")
    print(f"Candidate model: {MODEL_DIR}")
    print(f"Device: {'NVIDIA GPU' if device == 0 else 'CPU'}")
    print("Loading candidate model...")
    classifier = pipeline(
        "image-classification",
        model=str(MODEL_DIR),
        device=device,
    )

    y_true: list[str] = []
    y_pred: list[str] = []
    per_image_results: list[dict[str, object]] = []
    skipped = 0

    for folder_name, expected_label in CATEGORY_MAP.items():
        category_dir = TEST_DIR / folder_name
        if not category_dir.is_dir():
            print(f"SKIPPING missing category: {folder_name}")
            continue

        images = sorted(
            path for path in category_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f"\n{folder_name}: {len(images)} images")

        for image_path in images:
            try:
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                predictions = classifier(image, top_k=3)
                if not predictions:
                    skipped += 1
                    continue

                predicted_label = str(predictions[0]["label"])
                top3_labels = [str(item["label"]) for item in predictions]
                y_true.append(expected_label)
                y_pred.append(predicted_label)
                per_image_results.append({
                    "category_folder": folder_name,
                    "image": image_path.name,
                    "actual_label": expected_label,
                    "predicted_label": predicted_label,
                    "top1_score_percent": round(float(predictions[0]["score"]) * 100, 2),
                    "top3_correct": expected_label in top3_labels,
                })
            except (OSError, UnidentifiedImageError, ValueError) as error:
                skipped += 1
                print(f"Skipped {image_path.name}: {error}")

    if not y_true:
        raise RuntimeError("No test images were evaluated. Check the dataset category folders.")

    target_labels = list(CATEGORY_MAP.values())
    top1_accuracy = accuracy_score(y_true, y_pred)
    top3_accuracy = sum(bool(row["top3_correct"]) for row in per_image_results) / len(per_image_results)
    macro_precision = precision_score(
        y_true, y_pred, labels=target_labels, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        y_true, y_pred, labels=target_labels, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_true, y_pred, labels=target_labels, average="macro", zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=target_labels,
        target_names=list(CATEGORY_MAP.keys()),
        zero_division=0,
    )

    summary = (
        f"Model directory: {MODEL_DIR}\n"
        f"Device: {'CUDA GPU' if device == 0 else 'CPU'}\n"
        f"Images evaluated: {len(y_true)}\n"
        f"Images skipped: {skipped}\n"
        f"Top-1 accuracy: {top1_accuracy:.4f}\n"
        f"Top-3 accuracy: {top3_accuracy:.4f}\n"
        f"Macro precision: {macro_precision:.4f}\n"
        f"Macro recall: {macro_recall:.4f}\n"
        f"Macro F1-score: {macro_f1:.4f}\n\n"
        "Evaluation is limited to the mapped PlantDoc categories and this test split.\n"
        "The category mapping for 'Corn leaf blight' is an assumption and needs verification.\n"
        "This result is not a guarantee of general real-world diagnostic accuracy.\n"
    )

    (REPORT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    (REPORT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    with (REPORT_DIR / "per_image_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=[
            "category_folder", "image", "actual_label", "predicted_label",
            "top1_score_percent", "top3_correct",
        ])
        writer.writeheader()
        writer.writerows(per_image_results)

    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    with (REPORT_DIR / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["actual / predicted"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + row.tolist())

    print("\n" + "=" * 58)
    print("PLANTCARE AI — FINE-TUNED MODEL TEST RESULTS")
    print("=" * 58)
    print(summary)
    print("Per-category classification report:")
    print(report)
    print(f"Reports saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
