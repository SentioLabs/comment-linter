"""Generate labeled training data from codebases using comment-lint --export-features.

Runs the comment-lint binary on a source directory, reads the JSONL feature output,
and assigns initial labels based on heuristic scores:
  - score >= 0.7 → superfluous (label=1)
  - score <= 0.3 → valuable (label=0)
  - 0.3 < score < 0.7 → uncertain (label=-1, needs manual review)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def label_from_score(score: float) -> tuple[int, str]:
    """Map a heuristic score to a label and source tag.

    Returns:
        (label, source) where label is 1 (superfluous), 0 (valuable),
        or -1 (uncertain), and source is the labeling method.
    """
    if score >= 0.7:
        return 1, "heuristic_auto"
    elif score <= 0.3:
        return 0, "heuristic_auto"
    else:
        return -1, "uncertain"


def generate_labeled_data(
    binary: str, directory: str, output: str | None = None
) -> dict[str, int]:
    """Run comment-lint and produce labeled JSONL.

    Args:
        binary: Path to the comment-lint binary.
        directory: Source directory to scan.
        output: Optional output file path. If None, writes to stdout.

    Returns:
        Dict with counts: total, superfluous, valuable, uncertain.
    """
    result = subprocess.run(
        [binary, "--export-features", directory],
        capture_output=True,
        text=True,
    )

    if result.returncode not in (0, 1):
        print(f"comment-lint failed (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    counts = {"total": 0, "superfluous": 0, "valuable": 0, "uncertain": 0}
    out = open(output, "w") if output else sys.stdout

    try:
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            score = record.get("heuristic_score", 0.0)

            label, source = label_from_score(score)
            record["label"] = label
            record["label_source"] = source

            out.write(json.dumps(record) + "\n")

            counts["total"] += 1
            if label == 1:
                counts["superfluous"] += 1
            elif label == 0:
                counts["valuable"] += 1
            else:
                counts["uncertain"] += 1
    finally:
        if output:
            out.close()

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate labeled training data from codebases using comment-lint"
    )
    parser.add_argument(
        "--binary", required=True, help="Path to the comment-lint binary"
    )
    parser.add_argument(
        "--dir", required=True, help="Source directory to scan for comments"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: stdout)",
    )
    args = parser.parse_args()

    counts = generate_labeled_data(args.binary, args.dir, args.output)

    print(
        f"Generated {counts['total']} labeled examples:",
        file=sys.stderr,
    )
    print(f"  Superfluous: {counts['superfluous']}", file=sys.stderr)
    print(f"  Valuable:    {counts['valuable']}", file=sys.stderr)
    print(f"  Uncertain:   {counts['uncertain']}", file=sys.stderr)


if __name__ == "__main__":
    main()
