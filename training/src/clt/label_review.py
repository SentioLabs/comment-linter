"""Interactive CLI for manually reviewing uncertain comment labels.

Reads JSONL data, presents uncertain comments (label=-1) for human review,
and saves updated labels incrementally.
"""

import argparse
import json
import sys


def display_sample(record: dict, index: int, total: int) -> None:
    """Display a comment record for review."""
    print(f"\n{'=' * 60}")
    print(f"  [{index + 1}/{total}]  {record.get('file', '?')}:{record.get('line', '?')}")
    print(f"  Language: {record.get('language', '?')}  Kind: {record.get('comment_kind', '?')}")
    print(f"{'=' * 60}")
    print(f"\n  Comment: {record.get('comment_text', '')}")
    print(f"\n  Heuristic score:      {record.get('heuristic_score', 0):.3f}")
    print(f"  Heuristic confidence: {record.get('heuristic_confidence', 0):.3f}")

    features = record.get("features", {})
    key_features = [
        "token_overlap_jaccard",
        "identifier_substring_ratio",
        "has_why_indicator",
        "has_external_ref",
        "imperative_verb_noun",
        "is_section_label",
    ]
    print("\n  Key features:")
    for key in key_features:
        val = features.get(key, 0.0)
        if isinstance(val, bool):
            print(f"    {key}: {val}")
        else:
            print(f"    {key}: {float(val):.3f}")

    print()


def review_labels(input_path: str, output_path: str) -> None:
    """Review uncertain labels interactively.

    Args:
        input_path: Path to input JSONL file.
        output_path: Path to output JSONL file with updated labels.
    """
    records = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    uncertain = [(i, r) for i, r in enumerate(records) if r.get("label") == -1]

    if not uncertain:
        print("No uncertain labels to review.")
        with open(output_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        return

    print(f"Found {len(uncertain)} uncertain samples to review.")
    print("Commands: [s]uperfluous  [v]aluable  [skip]  [q]uit\n")

    reviewed = 0
    for idx, (orig_idx, record) in enumerate(uncertain):
        display_sample(record, idx, len(uncertain))

        while True:
            try:
                choice = input("  Label? [s/v/skip/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nSaving progress...", file=sys.stderr)
                choice = "q"

            if choice in ("s", "superfluous"):
                records[orig_idx]["label"] = 1
                records[orig_idx]["label_source"] = "manual_review"
                reviewed += 1
                break
            elif choice in ("v", "valuable"):
                records[orig_idx]["label"] = 0
                records[orig_idx]["label_source"] = "manual_review"
                reviewed += 1
                break
            elif choice == "skip":
                break
            elif choice in ("q", "quit"):
                print(f"\nReviewed {reviewed} comments. Saving...", file=sys.stderr)
                with open(output_path, "w") as f:
                    for r in records:
                        f.write(json.dumps(r) + "\n")
                return
            else:
                print("  Invalid choice. Use s, v, skip, or q.", file=sys.stderr)

        # Save incrementally
        with open(output_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    print(f"\nReview complete. Reviewed {reviewed} comments.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review uncertain labels interactively"
    )
    parser.add_argument(
        "--input", required=True, help="Input JSONL file with labeled data"
    )
    parser.add_argument(
        "--output", required=True, help="Output JSONL file with reviewed labels"
    )
    args = parser.parse_args()

    review_labels(args.input, args.output)


if __name__ == "__main__":
    main()
