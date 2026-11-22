
from pathlib import Path
import csv

import torch
from PIL import Image, UnidentifiedImageError
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# Use the real-world test images already downloaded.
PROJECT_DIR = Path(__file__).parent
TEST_DIR = PROJECT_DIR / "datasets" / "PlantDoc-test-only" / "test"
REPORT_DIR = PROJECT_DIR / "evaluation_results"

MODEL_NAME = "neghani/plantvillage-mobilenetv2"
MODEL_REVISION = "v1.0"

# Map only categories with a reasonably specific disease label.
CATEGORY_MAP = {
    "Corn Gray leaf spot":
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",

    "Corn leaf blight":
        "Corn_(maize)___Northern_Leaf_Blight",

    "Corn rust leaf":
        "Corn_(maize)___Common_rust_",

    "grape leaf black rot":
        "Grape___Black_rot",

    "Potato leaf early blight":
        "Potato___Early_blight",

    "Potato leaf late blight":
        "Potato___Late_blight",

    "Squash Powdery mildew leaf":
        "Squash___Powdery_mildew",

    "Tomato Early blight leaf":
        "Tomato___Early_blight",

    "Tomato leaf bacterial spot":
        "Tomato___Bacterial_spot",

    "Tomato leaf late blight":
        "Tomato___Late_blight",

    "Tomato leaf mosaic virus":
        "Tomato___Tomato_mosaic_virus",

    "Tomato leaf yellow virus":
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",

    "Tomato mold leaf":
        "Tomato___Leaf_Mold",

    "Tomato Septoria leaf spot":
        "Tomato___Septoria_leaf_spot",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}\n"
            "Check the dataset folder path."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    y_true = []
    y_pred = []
    per_image_results = []
    skipped = 0

    device = 0 if torch.cuda.is_available() else -1

    print(f"Test folder: {TEST_DIR}")
    print(f"Model: {MODEL_NAME}, revision {MODEL_REVISION}")
    print(f"Device: {'NVIDIA GPU' if device == 0 else 'CPU'}")
    print("Loading model...")

    classifier = pipeline(
        "image-classification",
        model=MODEL_NAME,
        revision=MODEL_REVISION,
        device=device,
    )

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

                predicted_label = predictions[0]["label"]
                top3_labels = [item["label"] for item in predictions]

                y_true.append(expected_label)
                y_pred.append(predicted_label)

                per_image_results.append({
                    "category_folder": folder_name,
                    "image": image_path.name,
                    "actual_label": expected_label,
                    "predicted_label": predicted_label,
                    "top1_score_percent": round(
                        float(predictions[0]["score"]) * 100, 2
                    ),
                    "top3_correct": expected_label in top3_labels,
                })

            except (OSError, UnidentifiedImageError, ValueError) as error:
                skipped += 1
                print(f"Skipped {image_path.name}: {error}")

    if not y_true:
        print("\nNo images were evaluated.")
        print("Check CATEGORY_MAP and your dataset folder names.")
        return

    accuracy = accuracy_score(y_true, y_pred)
    top3_accuracy = sum(
        row["top3_correct"] for row in per_image_results
    ) / len(per_image_results)

    macro_precision = precision_score(
        y_true, y_pred, labels=list(CATEGORY_MAP.values()),
        average="macro", zero_division=0
    )
    macro_recall = recall_score(
        y_true, y_pred, labels=list(CATEGORY_MAP.values()),
        average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_true, y_pred, labels=list(CATEGORY_MAP.values()),
        average="macro", zero_division=0
    )

    print("\n" + "=" * 55)
    print("PLANTCARE AI — PLANTDOC EVALUATION RESULTS")
    print("=" * 55)
    print(f"Images evaluated: {len(y_true)}")
    print(f"Images skipped: {skipped}")
    print(f"Top-1 accuracy: {accuracy:.2%}")
    print(f"Top-3 accuracy: {top3_accuracy:.2%}")
    print(f"Macro precision: {macro_precision:.2%}")
    print(f"Macro recall: {macro_recall:.2%}")
    print(f"Macro F1-score: {macro_f1:.2%}")

    print("\nPer-category classification report:")
    report = classification_report(
        y_true,
        y_pred,
        labels=list(CATEGORY_MAP.values()),
        target_names=list(CATEGORY_MAP.keys()),
        zero_division=0,
    )
    print(report)

    (REPORT_DIR / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    with (REPORT_DIR / "per_image_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "category_folder",
                "image",
                "actual_label",
                "predicted_label",
                "top1_score_percent",
                "top3_correct",
            ],
        )
        writer.writeheader()
        writer.writerows(per_image_results)

    # Include unexpected predicted labels in the confusion matrix.
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    with (REPORT_DIR / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["actual / predicted"] + labels)

        for label, row in zip(labels, matrix):
            writer.writerow([label] + row.tolist())

    summary = (
        f"Model: {MODEL_NAME}\n"
        f"Revision: {MODEL_REVISION}\n"
        f"Device: {'CUDA GPU' if device == 0 else 'CPU'}\n"
        f"Images evaluated: {len(y_true)}\n"
        f"Images skipped: {skipped}\n"
        f"Top-1 accuracy: {accuracy:.4f}\n"
        f"Top-3 accuracy: {top3_accuracy:.4f}\n"
        f"Macro precision: {macro_precision:.4f}\n"
        f"Macro recall: {macro_recall:.4f}\n"
        f"Macro F1-score: {macro_f1:.4f}\n"
        "\nEvaluation is limited to the mapped categories and this test set.\n"
        "It does not establish general real-world diagnostic accuracy.\n"
    )

    (REPORT_DIR / "summary.txt").write_text(summary, encoding="utf-8")

    print(f"\nReports saved in: {REPORT_DIR}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
