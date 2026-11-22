from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "datasets" / "PlantDoc-test-only"
TRAIN_DIR = DATASET_DIR / "train"
MODEL_OUTPUT_DIR = PROJECT_DIR / "models" / "plantdoc_finetuned"
RESULTS_DIR = PROJECT_DIR / "training_results"

BASE_MODEL = "neghani/plantvillage-mobilenetv2"
BASE_REVISION = "v1.0"
SEED = 42
EPOCHS = 6
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
VALIDATION_FRACTION = 0.20
EARLY_STOPPING_PATIENCE = 2
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# These names match the test-category mapping used for the initial benchmark.
# Note: "Corn leaf blight" -> Northern_Leaf_Blight is an assumption, not a
# verified plant-pathology label. Do not present this category's metrics as
# definitive until its dataset annotation has been checked.
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


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_image_paths(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class PlantDocDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], training: bool = False):
        self.samples = samples
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        image_path, label_id = self.samples[index]
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Cannot read image {image_path}: {exc}") from exc

        # Lightweight augmentations help the model see varied orientations and
        # lighting while preserving the original image files.
        if self.training:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            if random.random() < 0.15:
                image = ImageOps.flip(image)
            if random.random() < 0.5:
                image = image.rotate(random.uniform(-18, 18), resample=Image.Resampling.BILINEAR)
            if random.random() < 0.35:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            if random.random() < 0.35:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
            if random.random() < 0.25:
                image = ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))

        return image, label_id


def make_collate_fn(processor: Any):
    def collate(batch: list[tuple[Image.Image, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        images = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        encoded = processor(images=images, return_tensors="pt")
        return encoded["pixel_values"], labels
    return collate


def split_training_images(label2id: dict[str, int]) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[dict[str, Any]]]:
    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {TRAIN_DIR}\n"
            "Run: git -C \".\\datasets\\PlantDoc-test-only\" sparse-checkout set test train"
        )

    training_samples: list[tuple[Path, int]] = []
    validation_samples: list[tuple[Path, int]] = []
    counts: list[dict[str, Any]] = []
    problems: list[str] = []

    for index, (folder_name, expected_label) in enumerate(CATEGORY_MAP.items()):
        if expected_label not in label2id:
            problems.append(f"Model label missing: {expected_label}")
            continue

        category_dir = TRAIN_DIR / folder_name
        if not category_dir.is_dir():
            problems.append(f"Training category folder missing: {category_dir}")
            continue

        images = get_image_paths(category_dir)
        if len(images) < 2:
            problems.append(f"Need at least 2 images in {category_dir}; found {len(images)}")
            continue

        random.Random(SEED + index).shuffle(images)
        validation_count = max(1, round(len(images) * VALIDATION_FRACTION))
        validation_count = min(validation_count, len(images) - 1)
        val_images = images[:validation_count]
        train_images = images[validation_count:]
        label_id = label2id[expected_label]

        training_samples.extend((path, label_id) for path in train_images)
        validation_samples.extend((path, label_id) for path in val_images)
        counts.append({
            "category_folder": folder_name,
            "target_label": expected_label,
            "train_images": len(train_images),
            "validation_images": len(val_images),
        })

    if problems:
        formatted = "\n - ".join(problems)
        raise RuntimeError(
            "Dataset/model preflight checks failed. No training was started.\n - " + formatted
        )

    return training_samples, validation_samples, counts


def evaluate_validation(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    targeted_label_ids: list[int],
) -> dict[str, float]:
    model.eval()
    actual: list[int] = []
    predicted: list[int] = []
    loss_sum = 0.0
    image_count = 0

    with torch.no_grad():
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            batch_size = labels.size(0)
            loss_sum += float(outputs.loss.item()) * batch_size
            image_count += batch_size
            predicted.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            actual.extend(labels.cpu().tolist())

    return {
        "validation_loss": loss_sum / max(image_count, 1),
        "validation_accuracy": float(accuracy_score(actual, predicted)),
        "validation_macro_f1": float(f1_score(
            actual, predicted,
            labels=targeted_label_ids,
            average="macro",
            zero_division=0,
        )),
        "validation_macro_recall": float(recall_score(
            actual, predicted,
            labels=targeted_label_ids,
            average="macro",
            zero_division=0,
        )),
    }


def main() -> None:
    set_seeds(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PLANTCARE AI — PLANTDOC FINE-TUNING")
    print("=" * 58)
    print(f"Dataset train directory: {TRAIN_DIR}")
    print(f"Base model: {BASE_MODEL} (revision {BASE_REVISION})")
    print(f"Compute device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print("Test images will not be used in training or validation.")
    print("Loading the pretrained model and image processor...")

    processor = AutoImageProcessor.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    model = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
    )

    id2label = {int(index): label for index, label in model.config.id2label.items()}
    label2id = {label: index for index, label in id2label.items()}
    missing_labels = [label for label in CATEGORY_MAP.values() if label not in label2id]
    if missing_labels:
        available = "\n".join(id2label[index] for index in sorted(id2label))
        raise RuntimeError(
            "Some expected PlantVillage labels are not present in the model config:\n"
            + "\n".join(missing_labels)
            + "\nLabels provided by the model:\n"
            + available
        )

    train_samples, validation_samples, counts = split_training_images(label2id)
    with (RESULTS_DIR / "dataset_counts.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "category_folder", "target_label", "train_images", "validation_images"
        ])
        writer.writeheader()
        writer.writerows(counts)

    print("\nImage counts by category:")
    for row in counts:
        print(f"  {row['category_folder']}: train={row['train_images']}, validation={row['validation_images']}")
    print(f"\nTotal training images: {len(train_samples)}")
    print(f"Total validation images: {len(validation_samples)}")
    print(f"Model classes preserved: {model.config.num_labels}")

    train_dataset = PlantDocDataset(train_samples, training=True)
    validation_dataset = PlantDocDataset(validation_samples, training=False)
    collate_fn = make_collate_fn(processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    targeted_label_ids = sorted({label for _, label in train_samples + validation_samples})
    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen_images = 0

        for step, (pixel_values, labels) in enumerate(train_loader, start=1):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            seen_images += batch_size

            if step % 20 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch}/{EPOCHS} | batch {step}/{len(train_loader)} | "
                    f"mean train loss {running_loss / max(seen_images, 1):.4f}"
                )

        scheduler.step()
        train_loss = running_loss / max(seen_images, 1)
        val_metrics = evaluate_validation(model, validation_loader, device, targeted_label_ids)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "learning_rate_next_epoch": scheduler.get_last_lr()[0],
        }
        history.append(row)
        with (RESULTS_DIR / "training_history.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        print(
            f"\nEpoch {epoch} summary: train_loss={train_loss:.4f}, "
            f"val_accuracy={val_metrics['validation_accuracy']:.2%}, "
            f"val_macro_F1={val_metrics['validation_macro_f1']:.2%}, "
            f"val_macro_recall={val_metrics['validation_macro_recall']:.2%}\n"
        )

        current_f1 = val_metrics["validation_macro_f1"]
        if current_f1 > best_macro_f1:
            best_macro_f1 = current_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save_pretrained(MODEL_OUTPUT_DIR, safe_serialization=True)
            processor.save_pretrained(MODEL_OUTPUT_DIR)
            metadata = {
                "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION,
                "best_epoch": best_epoch,
                "best_validation_macro_f1": best_macro_f1,
                "best_validation_metrics": val_metrics,
                "train_images": len(train_samples),
                "validation_images": len(validation_samples),
                "seed": SEED,
                "classes_preserved": model.config.num_labels,
                "category_mapping": CATEGORY_MAP,
                "notes": [
                    "Candidate model only; not accepted for deployment until held-out test comparison.",
                    "The mapped Corn leaf blight label is an assumption requiring annotation verification.",
                    "PlantDoc test images were not used during training or validation.",
                ],
            }
            (MODEL_OUTPUT_DIR / "plantcare_training_metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            print(f"Saved new best candidate to: {MODEL_OUTPUT_DIR}")
        else:
            epochs_without_improvement += 1
            print(f"Validation macro F1 did not improve ({epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}).")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping: validation macro F1 has not improved.")
            break

    final_summary = {
        "candidate_model_directory": str(MODEL_OUTPUT_DIR),
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "train_images": len(train_samples),
        "validation_images": len(validation_samples),
        "test_images_used_during_training": 0,
        "status": "Candidate saved; held-out test evaluation is still required.",
    }
    (RESULTS_DIR / "training_summary.json").write_text(
        json.dumps(final_summary, indent=2), encoding="utf-8"
    )
    print("\n" + "=" * 58)
    print("FINE-TUNING COMPLETE")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro F1: {best_macro_f1:.2%}")
    print(f"Candidate model: {MODEL_OUTPUT_DIR}")
    print(f"Training logs: {RESULTS_DIR}")
    print("Next: run evaluate_plantdoc_finetuned.py on the held-out test set.")


if __name__ == "__main__":
    main()
