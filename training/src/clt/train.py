"""Train binary classifiers for comment superfluousness detection.

Loads JSONL training data, extracts features in the exact order matching
tensor.rs, trains multiple classifiers, and saves the best model via joblib.

Usage:
    python train.py --train train.jsonl [--val val.jsonl] --output model.joblib
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from clt.utils import BOOL_FEATURES, FEATURE_DIM, FEATURE_NAMES, load_jsonl


def _encode_feature(name: str, value) -> float:
    """Encode a single feature value to float32.

    Args:
        name: Feature name.
        value: Raw feature value from JSONL.

    Returns:
        Float-encoded feature value.
    """
    if value is None:
        return 0.0
    if name in BOOL_FEATURES:
        return 1.0 if value else 0.0
    return float(value)


def extract_features(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and label vector y from JSONL records.

    Features are extracted in the exact order defined by FEATURE_NAMES,
    which matches tensor.rs.

    Args:
        records: List of JSONL records with "features" and "label" keys.

    Returns:
        Tuple of (X, y) where X is shape (n, 16) float32 and y is shape (n,) int.
    """
    X = np.zeros((len(records), FEATURE_DIM), dtype=np.float32)
    y = np.zeros(len(records), dtype=np.int32)

    for i, record in enumerate(records):
        features = record["features"]
        for j, name in enumerate(FEATURE_NAMES):
            raw_value = features.get(name)
            X[i, j] = _encode_feature(name, raw_value)
        y[i] = record["label"]

    return X, y


def train_models(
    X: np.ndarray, y: np.ndarray
) -> dict[str, object]:
    """Train multiple classifiers on the given features and labels.

    Args:
        X: Feature matrix of shape (n, 16).
        y: Label vector of shape (n,).

    Returns:
        Dictionary mapping model name to trained model instance.
    """
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, solver="lbfgs",
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1,
            class_weight="balanced",
        ),
        "XGBClassifier": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=float((y == 0).sum()) / max(float((y == 1).sum()), 1),
        ),
    }

    for name, model in models.items():
        model.fit(X, y)

    return models


def select_best_model(
    models: dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> tuple[str, object, dict]:
    """Select the best model using cross-validated F1 score.

    Args:
        models: Dictionary of trained models.
        X: Feature matrix.
        y: Label vector.
        cv: Number of cross-validation folds.

    Returns:
        Tuple of (best_model_name, best_model, scores_dict).
    """
    scores = {}
    best_name = None
    best_f1 = -1.0

    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
        mean_f1 = cv_scores.mean()
        scores[name] = {
            "f1_mean": mean_f1,
            "f1_std": cv_scores.std(),
            "f1_scores": cv_scores.tolist(),
        }
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_name = name

    return best_name, models[best_name], scores


def report_cross_validation(
    models: dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> None:
    """Run cross-validation and print results for all models.

    Args:
        models: Dictionary of trained models.
        X: Feature matrix.
        y: Label vector.
        cv: Number of cross-validation folds.
    """
    scoring_metrics = ["accuracy", "precision", "recall", "f1"]

    for name, model in models.items():
        print(f"\n--- {name} ---")
        for metric in scoring_metrics:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            print(f"  {metric:>10s}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


def main() -> None:
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train binary classifiers for comment superfluousness detection"
    )
    parser.add_argument(
        "--train",
        required=True,
        nargs="+",
        help="Path(s) to training JSONL files (merged at runtime)",
    )
    parser.add_argument(
        "--val",
        default=None,
        nargs="+",
        help="Path(s) to validation JSONL files (optional)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the best model (joblib format)",
    )
    args = parser.parse_args()

    # Load training data (merge multiple sources)
    train_records = []
    for path in args.train:
        records = load_jsonl(path)
        print(f"Loaded {len(records)} records from {path}", file=sys.stderr)
        train_records.extend(records)
    print(f"Total training records: {len(train_records)}", file=sys.stderr)

    X_train, y_train = extract_features(train_records)
    print(
        f"  Features: {X_train.shape[1]}, "
        f"Positive: {(y_train == 1).sum()}, "
        f"Negative: {(y_train == 0).sum()}",
        file=sys.stderr,
    )

    # Train models
    print("\nTraining models...", file=sys.stderr)
    models = train_models(X_train, y_train)

    # Cross-validation report
    print("\nCross-validation results:", file=sys.stderr)
    report_cross_validation(models, X_train, y_train)

    # Select best model
    best_name, best_model, scores = select_best_model(models, X_train, y_train)
    print(
        f"\nBest model: {best_name} "
        f"(F1={scores[best_name]['f1_mean']:.4f})",
        file=sys.stderr,
    )

    # Evaluate on validation set if provided
    if args.val:
        val_records = []
        for path in args.val:
            records = load_jsonl(path)
            print(f"Loaded {len(records)} val records from {path}", file=sys.stderr)
            val_records.extend(records)
        X_val, y_val = extract_features(val_records)
        from sklearn.metrics import classification_report

        y_pred = best_model.predict(X_val)
        print(classification_report(y_val, y_pred), file=sys.stderr)

    # Save best model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, str(output_path))
    print(f"\nSaved best model ({best_name}) to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
