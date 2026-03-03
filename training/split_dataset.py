"""Split labeled JSONL data into train/validation/test sets.

Performs stratified splitting to maintain class balance across splits.
Excludes uncertain (label=-1) examples from the output.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


def stratified_split(
    records: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into train/val/test with stratification by label.

    Args:
        records: List of labeled records (must have 'label' field).
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        seed: Random seed for reproducibility.

    Returns:
        (train, val, test) lists of records.
    """
    rng = random.Random(seed)

    by_label: dict[int, list[dict]] = {}
    for record in records:
        label = record["label"]
        by_label.setdefault(label, []).append(record)

    train, val, test = [], [], []

    for label, group in sorted(by_label.items()):
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio)) if n > 2 else 0
        n_test = n - n_train - n_val

        if n_test < 0:
            n_val = n - n_train
            n_test = 0

        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def write_jsonl(records: list[dict], path: Path) -> None:
    """Write records as JSONL to a file."""
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def report_distribution(name: str, records: list[dict]) -> None:
    """Print label distribution for a split."""
    counts = Counter(r["label"] for r in records)
    total = len(records)
    print(f"  {name}: {total} examples")
    for label in sorted(counts.keys()):
        pct = counts[label] / total * 100 if total > 0 else 0
        print(f"    label={label}: {counts[label]} ({pct:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test sets with stratified splitting"
    )
    parser.add_argument(
        "--input", required=True, help="Input JSONL file with labeled data"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for split files",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.70, help="Training set ratio (default: 0.70)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    records = []
    skipped = 0
    with open(args.input) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("label") == -1:
                skipped += 1
                continue
            records.append(record)

    if not records:
        print("No labeled examples found (all uncertain?).")
        sys.exit(1)

    print(f"Loaded {len(records)} labeled examples ({skipped} uncertain excluded)")

    train, val, test = stratified_split(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(train, output_dir / "train.jsonl")
    write_jsonl(val, output_dir / "val.jsonl")
    write_jsonl(test, output_dir / "test.jsonl")

    print("\nSplit distribution:")
    report_distribution("train", train)
    report_distribution("val", val)
    report_distribution("test", test)


if __name__ == "__main__":
    main()
