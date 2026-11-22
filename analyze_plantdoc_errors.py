from pathlib import Path
import csv
from collections import Counter, defaultdict

PROJECT_DIR = Path(__file__).parent
CSV_PATH = PROJECT_DIR / "evaluation_results" / "per_image_predictions.csv"


def main():
    if not CSV_PATH.exists():
        print(f"Prediction file not found: {CSV_PATH}")
        print("Run python evaluate_plantdoc.py first.")
        return

    categories = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "wrong_predictions": Counter(),
    })

    all_predictions = Counter()
    correct_scores = []
    incorrect_scores = []
    total = 0

    with CSV_PATH.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        for row in csv.DictReader(file):
            category = row["category_folder"]
            actual = row["actual_label"]
            predicted = row["predicted_label"]
            score = float(row["top1_score_percent"])

            total += 1
            categories[category]["total"] += 1
            all_predictions[predicted] += 1

            if actual == predicted:
                categories[category]["correct"] += 1
                correct_scores.append(score)
            else:
                categories[category]["wrong_predictions"][predicted] += 1
                incorrect_scores.append(score)

    print("\nPLANTCARE AI — ERROR ANALYSIS")
    print("=" * 65)
    print(f"Total images: {total}")

    for category, data in sorted(categories.items()):
        count = data["total"]
        correct = data["correct"]
        recall = correct / count * 100 if count else 0

        print(f"\n{category}")
        print(f"  Correct: {correct}/{count} ({recall:.1f}%)")

        wrong = data["wrong_predictions"].most_common(3)
        if wrong:
            print("  Most common incorrect predictions:")
            for label, number in wrong:
                print(f"    {label}: {number}")

    print("\nMOST FREQUENTLY PREDICTED LABELS")
    print("=" * 65)
    for label, count in all_predictions.most_common(15):
        print(f"{label}: {count}")

    if correct_scores:
        print(
            "\nAverage confidence on correct predictions: "
            f"{sum(correct_scores) / len(correct_scores):.2f}%"
        )

    if incorrect_scores:
        print(
            "Average confidence on incorrect predictions: "
            f"{sum(incorrect_scores) / len(incorrect_scores):.2f}%"
        )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
    