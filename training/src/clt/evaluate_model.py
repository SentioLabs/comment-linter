"""Evaluate an ONNX model on test data and compare with heuristic scores.

Loads an ONNX model and test JSONL, runs inference, and reports
accuracy, precision, recall, F1, and confusion matrix.

Usage:
    python evaluate_model.py --model model.onnx --test test.jsonl
"""

import argparse
import json
import sys

import numpy as np
import onnxruntime as ort
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from clt.utils import FEATURE_DIM, FEATURE_NAMES, load_jsonl
from clt.train import extract_features


def load_onnx_session(model_path: str) -> ort.InferenceSession:
    """Load an ONNX model and return an inference session.

    Args:
        model_path: Path to the ONNX model file.

    Returns:
        ONNX Runtime InferenceSession.
    """
    return ort.InferenceSession(model_path)


def predict_onnx(
    session: ort.InferenceSession, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on an ONNX model.

    Args:
        session: ONNX Runtime session.
        X: Feature matrix of shape (n, 16), dtype float32.

    Returns:
        Tuple of (predicted_labels, probabilities) where
        predicted_labels is shape (n,) and probabilities is shape (n, 2).
    """
    X = X.astype(np.float32)
    results = session.run(["probabilities"], {"input": X})
    probs = results[0]
    # prob[0] = P(not superfluous), prob[1] = P(superfluous)
    predicted = (probs[:, 1] >= 0.5).astype(np.int32)
    return predicted, probs


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
) -> dict:
    """Compute evaluation metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        probs: Prediction probabilities of shape (n, 2).

    Returns:
        Dictionary of evaluation metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def compare_with_heuristic(
    records: list[dict],
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Compare model predictions with heuristic scores.

    Args:
        records: Original JSONL records (may contain heuristic_score).
        y_pred: Model-predicted labels.
        y_true: True labels.

    Returns:
        Dictionary with comparison metrics.
    """
    heuristic_preds = []
    valid_indices = []

    for i, record in enumerate(records):
        score = record.get("heuristic_score")
        if score is not None:
            heuristic_pred = 1 if score >= 0.5 else 0
            heuristic_preds.append(heuristic_pred)
            valid_indices.append(i)

    if not heuristic_preds:
        return {"heuristic_available": False}

    heuristic_preds = np.array(heuristic_preds)
    y_true_subset = y_true[valid_indices]
    y_pred_subset = y_pred[valid_indices]

    return {
        "heuristic_available": True,
        "n_with_heuristic": len(valid_indices),
        "heuristic_accuracy": accuracy_score(y_true_subset, heuristic_preds),
        "heuristic_f1": f1_score(y_true_subset, heuristic_preds, zero_division=0),
        "model_accuracy": accuracy_score(y_true_subset, y_pred_subset),
        "model_f1": f1_score(y_true_subset, y_pred_subset, zero_division=0),
    }


def main() -> None:
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate an ONNX model on test data"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the ONNX model file",
    )
    parser.add_argument(
        "--test",
        required=True,
        help="Path to the test JSONL file",
    )
    args = parser.parse_args()

    # Load model
    print(f"Loading ONNX model from {args.model}...", file=sys.stderr)
    session = load_onnx_session(args.model)

    # Load test data
    print(f"Loading test data from {args.test}...", file=sys.stderr)
    records = load_jsonl(args.test)
    print(f"  Loaded {len(records)} records", file=sys.stderr)

    X, y_true = extract_features(records)

    # Run inference
    y_pred, probs = predict_onnx(session, X)

    # Report metrics
    print("\n=== Model Evaluation ===", file=sys.stderr)
    metrics = evaluate(y_true, y_pred, probs)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}", file=sys.stderr)
    print(f"  Precision: {metrics['precision']:.4f}", file=sys.stderr)
    print(f"  Recall:    {metrics['recall']:.4f}", file=sys.stderr)
    print(f"  F1:        {metrics['f1']:.4f}", file=sys.stderr)
    print(f"\nConfusion Matrix:", file=sys.stderr)
    cm = np.array(metrics["confusion_matrix"])
    print(f"  TN={cm[0][0]}, FP={cm[0][1]}", file=sys.stderr)
    print(f"  FN={cm[1][0]}, TP={cm[1][1]}", file=sys.stderr)

    # Detailed classification report
    print("\nClassification Report:", file=sys.stderr)
    print(
        classification_report(
            y_true, y_pred,
            target_names=["valuable (0)", "superfluous (1)"],
        ),
        file=sys.stderr,
    )

    # Compare with heuristic
    comparison = compare_with_heuristic(records, y_pred, y_true)
    if comparison.get("heuristic_available"):
        print("\n=== Heuristic Comparison ===", file=sys.stderr)
        print(
            f"  Samples with heuristic: {comparison['n_with_heuristic']}",
            file=sys.stderr,
        )
        print(
            f"  Heuristic Accuracy: {comparison['heuristic_accuracy']:.4f}",
            file=sys.stderr,
        )
        print(
            f"  Heuristic F1:       {comparison['heuristic_f1']:.4f}",
            file=sys.stderr,
        )
        print(
            f"  Model Accuracy:     {comparison['model_accuracy']:.4f}",
            file=sys.stderr,
        )
        print(
            f"  Model F1:           {comparison['model_f1']:.4f}",
            file=sys.stderr,
        )
    else:
        print(
            "\nNo heuristic scores found in test data for comparison.",
            file=sys.stderr,
        )

    # Output metrics as JSON to stdout for programmatic use
    output = {
        "metrics": metrics,
        "heuristic_comparison": comparison,
        "n_samples": len(records),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
