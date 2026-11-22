from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "datasets" / "PlantDoc-test-only"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = PROJECT_DIR / "dataset_audit"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Match the 14 categories in the existing baseline and fine-tuning scripts.
# The Corn leaf blight -> Northern Leaf Blight mapping is an assumption and
# must be checked against the dataset's source annotation before reporting it
# as disease-specific ground truth.
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


def list_images(category_dir: Path) -> list[Path]:
    if not category_dir.is_dir():
        return []
    return sorted(
        path for path in category_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def decoded_pixel_hash(path: Path) -> str:
    """Hash decoded pixels, ignoring file metadata and EXIF orientation differences."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"{image.width}x{image.height}:RGB:".encode("ascii"))
        digest.update(image.tobytes())
        return digest.hexdigest()


def inspect_image(path: Path) -> tuple[bool, str, str | None]:
    try:
        with Image.open(path) as source:
            source.verify()
        pixel_hash = decoded_pixel_hash(path)
        return True, "", pixel_hash
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        return False, str(exc), None


def main() -> None:
    if not TRAIN_DIR.is_dir() or not TEST_DIR.is_dir():
        raise FileNotFoundError(
            "Both train and test folders are required.\n"
            f"Expected training folder: {TRAIN_DIR}\n"
            f"Expected test folder: {TEST_DIR}\n"
            'Run: git -C ".\\datasets\\PlantDoc-test-only" sparse-checkout set test train'
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    category_rows: list[dict[str, Any]] = []
    unreadable_rows: list[dict[str, str]] = []
    duplicates: list[dict[str, str]] = []
    hash_locations: dict[str, list[dict[str, str]]] = defaultdict(list)
    missing_categories: list[str] = []
    total_train = total_test = 0
    valid_train = valid_test = 0

    print("PLANTCARE AI — DATASET INTEGRITY AUDIT")
    print("=" * 72)
    print(f"Dataset: {DATASET_DIR}")
    print("This audit reads images only; it does not modify them or train a model.\n")

    for category in CATEGORY_MAP:
        train_category = TRAIN_DIR / category
        test_category = TEST_DIR / category
        train_images = list_images(train_category)
        test_images = list_images(test_category)
        if not train_category.is_dir() or not test_category.is_dir():
            missing_categories.append(category)

        category_valid_train = 0
        category_valid_test = 0
        for split, images in (("train", train_images), ("test", test_images)):
            for image_path in images:
                okay, error, pixel_hash = inspect_image(image_path)
                if split == "train":
                    total_train += 1
                    if okay:
                        valid_train += 1
                        category_valid_train += 1
                else:
                    total_test += 1
                    if okay:
                        valid_test += 1
                        category_valid_test += 1
                if not okay:
                    unreadable_rows.append({
                        "split": split,
                        "category_folder": category,
                        "image": str(image_path.relative_to(DATASET_DIR)),
                        "error": error,
                    })
                elif pixel_hash:
                    hash_locations[pixel_hash].append({
                        "split": split,
                        "category_folder": category,
                        "image": str(image_path.relative_to(DATASET_DIR)),
                    })

        category_rows.append({
            "category_folder": category,
            "target_model_label": CATEGORY_MAP[category],
            "train_images": len(train_images),
            "train_valid_images": category_valid_train,
            "test_images": len(test_images),
            "test_valid_images": category_valid_test,
            "train_folder_exists": train_category.is_dir(),
            "test_folder_exists": test_category.is_dir(),
        })
        print(
            f"{category}: train {category_valid_train}/{len(train_images)} valid; "
            f"test {category_valid_test}/{len(test_images)} valid"
        )

    for occurrences in hash_locations.values():
        if len(occurrences) > 1:
            splits = {item["split"] for item in occurrences}
            duplicate_type = "cross_split" if len(splits) > 1 else "within_split"
            for item in occurrences:
                duplicates.append({
                    "duplicate_type": duplicate_type,
                    "split": item["split"],
                    "category_folder": item["category_folder"],
                    "image": item["image"],
                    "matching_images": " | ".join(
                        other["image"] for other in occurrences if other["image"] != item["image"]
                    ),
                })

    all_train_dirs = sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())
    all_test_dirs = sorted(p.name for p in TEST_DIR.iterdir() if p.is_dir())
    unmapped_train = [name for name in all_train_dirs if name not in CATEGORY_MAP]
    unmapped_test = [name for name in all_test_dirs if name not in CATEGORY_MAP]

    cross_split_duplicate_groups: dict[str, int] = {}
    for occurrences in hash_locations.values():
        if len({item["split"] for item in occurrences}) > 1:
            cross_split_duplicate_groups[occurrences[0]["image"]] = len(occurrences)

    with (OUTPUT_DIR / "per_category.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(category_rows[0].keys()))
        writer.writeheader()
        writer.writerows(category_rows)

    with (OUTPUT_DIR / "unreadable_images.csv").open("w", newline="", encoding="utf-8-sig") as file:
        fields = ["split", "category_folder", "image", "error"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(unreadable_rows)

    with (OUTPUT_DIR / "duplicate_images.csv").open("w", newline="", encoding="utf-8-sig") as file:
        fields = ["duplicate_type", "split", "category_folder", "image", "matching_images"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(duplicates)

    summary_lines = [
        "PLANTCARE AI — DATASET INTEGRITY AUDIT",
        "=" * 72,
        f"Dataset root: {DATASET_DIR}",
        f"Mapped categories checked: {len(CATEGORY_MAP)}",
        f"Training images found: {total_train}",
        f"Valid training images: {valid_train}",
        f"Test images found: {total_test}",
        f"Valid test images: {valid_test}",
        f"Unreadable images: {len(unreadable_rows)}",
        f"Duplicate-file pixel matches listed: {len(duplicates)}",
        f"Cross-split duplicate groups: {len(cross_split_duplicate_groups)}",
        f"Missing mapped category folders: {len(missing_categories)}",
        f"Unmapped train category folders (not used by current experiment): {len(unmapped_train)}",
        f"Unmapped test category folders (not used by current experiment): {len(unmapped_test)}",
        "",
        "MISSING MAPPED CATEGORIES:",
        *([f" - {name}" for name in missing_categories] or [" - None"]),
        "",
        "UNMAPPED TRAIN FOLDERS:",
        *([f" - {name}" for name in unmapped_train] or [" - None"]),
        "",
        "UNMAPPED TEST FOLDERS:",
        *([f" - {name}" for name in unmapped_test] or [" - None"]),
        "",
        "REVIEW NOTES:",
        " - Pixel hashes detect identical decoded pixels, not all resized or recompressed copies.",
        " - Duplicate groups spanning train and test should be reviewed before using test metrics.",
        " - Folder-to-disease mapping must be checked against dataset annotation; the Corn leaf blight mapping is currently an assumption.",
        " - This audit checks file integrity and split overlap; it cannot establish that every folder label is biologically correct.",
    ]
    (OUTPUT_DIR / "audit_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"Training: {valid_train}/{total_train} valid images")
    print(f"Test: {valid_test}/{total_test} valid images")
    print(f"Unreadable images: {len(unreadable_rows)}")
    print(f"Duplicate pixel matches listed: {len(duplicates)}")
    print(f"Cross-split duplicate groups: {len(cross_split_duplicate_groups)}")
    if missing_categories:
        print("WARNING: Some mapped category folders are missing.")
    if unreadable_rows:
        print("WARNING: Some image files could not be read. Review unreadable_images.csv.")
    if cross_split_duplicate_groups:
        print("WARNING: Identical decoded images appear in both train and test; review before trusting test metrics.")
    if unmapped_train or unmapped_test:
        print("NOTE: Some dataset categories are not part of this 14-category experiment.")
    print(f"Audit reports saved to: {OUTPUT_DIR}")
    print("Audit complete.")


if __name__ == "__main__":
    main()
