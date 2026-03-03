"""Export a trained scikit-learn or XGBoost model to ONNX format.

The exported ONNX model conforms to the MLScorer contract:
  - Input: name="input", shape=[None, 16], dtype=float32
  - Output: name="probabilities", shape=[None, 2], dtype=float32

Usage:
    python export_onnx.py --model model.joblib --output model.onnx
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from clt.utils import FEATURE_DIM, OUTPUT_DIM


def _is_xgboost_model(model) -> bool:
    """Check if the model is an XGBoost classifier."""
    return isinstance(model, XGBClassifier)


def _is_sklearn_model(model) -> bool:
    """Check if the model is a scikit-learn classifier."""
    return isinstance(model, (LogisticRegression, RandomForestClassifier))


def _convert_sklearn_to_onnx(model, input_name: str = "input") -> onnx.ModelProto:
    """Convert a scikit-learn model to ONNX format.

    Uses zipmap=False to ensure output is a tensor rather than a dictionary,
    which is needed for consistent output shape [None, 2].

    Args:
        model: Trained scikit-learn classifier.
        input_name: Name for the input tensor.

    Returns:
        ONNX ModelProto.
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [(input_name, FloatTensorType([None, FEATURE_DIM]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=13,
        options={type(model): {"zipmap": False}},
    )
    return onnx_model


def _convert_xgboost_to_onnx(model, input_name: str = "input") -> onnx.ModelProto:
    """Convert an XGBoost model to ONNX format.

    Uses skl2onnx with XGBoost converter registration. Falls back to
    converting via sklearn wrapper if direct conversion fails.

    Args:
        model: Trained XGBClassifier.
        input_name: Name for the input tensor.

    Returns:
        ONNX ModelProto.
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    # Ensure XGBoost converters are registered with skl2onnx
    try:
        from skl2onnx import update_registered_converter
        from skl2onnx.common.shape_calculator import (
            calculate_linear_classifier_output_shapes,
        )
        from skl2onnx.operator_converters.ada_boost import _apply_zipmap
    except ImportError:
        pass  # Converter may already be registered

    initial_type = [(input_name, FloatTensorType([None, FEATURE_DIM]))]

    # Try direct conversion first (skl2onnx may already know XGBClassifier)
    try:
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=13,
            options={type(model): {"zipmap": False}},
        )
        return onnx_model
    except Exception:
        # Fall back to conversion without options
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=13,
        )
        return onnx_model


def _rename_outputs(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """Rename ONNX model outputs to match the MLScorer contract.

    The model must have a probability output that gets renamed to "probabilities".
    The label output (if present) is left as-is.

    Args:
        onnx_model: ONNX model to modify.

    Returns:
        Modified ONNX model with correct output names.
    """
    graph = onnx_model.graph

    # Find the probability output - skl2onnx typically names it "probabilities"
    # or "output_probability". We need to find it and rename to "probabilities".
    prob_output = None
    prob_output_idx = None
    for idx, output in enumerate(graph.output):
        name_lower = output.name.lower()
        if "probab" in name_lower or "prob" in name_lower:
            prob_output = output
            prob_output_idx = idx
            break

    if prob_output is None:
        # If no probability output found, look for the second output
        # (skl2onnx typically outputs [label, probabilities])
        if len(graph.output) >= 2:
            prob_output = graph.output[1]
            prob_output_idx = 1
        elif len(graph.output) == 1:
            prob_output = graph.output[0]
            prob_output_idx = 0

    if prob_output is None:
        raise ValueError("Could not find probability output in ONNX model")

    old_name = prob_output.name
    new_name = "probabilities"

    if old_name == new_name:
        # Already named correctly; just ensure we only keep this output
        pass
    else:
        # Rename the output in the graph
        # Update all nodes that reference this output
        for node in graph.node:
            for i, output_name in enumerate(node.output):
                if output_name == old_name:
                    node.output[i] = new_name

        prob_output.name = new_name

    # Remove non-probability outputs (we only want "probabilities")
    outputs_to_keep = []
    for output in graph.output:
        if output.name == "probabilities":
            outputs_to_keep.append(output)

    # Clear and re-add outputs
    while len(graph.output) > 0:
        graph.output.pop()
    for output in outputs_to_keep:
        graph.output.append(output)

    return onnx_model


def _validate_onnx(onnx_path: str) -> None:
    """Validate the exported ONNX model against the MLScorer contract.

    Args:
        onnx_path: Path to the ONNX file.

    Raises:
        ValueError: If the model does not match the expected contract.
    """
    # Load and check with ONNX
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Load with ONNX Runtime and verify input/output
    session = ort.InferenceSession(onnx_path)

    # Check input
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"Expected 1 input, got {len(inputs)}")
    if inputs[0].name != "input":
        raise ValueError(f"Expected input name 'input', got '{inputs[0].name}'")
    if inputs[0].shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected input dim {FEATURE_DIM}, got {inputs[0].shape[1]}"
        )

    # Check output
    outputs = session.get_outputs()
    output_names = [o.name for o in outputs]
    if "probabilities" not in output_names:
        raise ValueError(
            f"Expected output 'probabilities', got {output_names}"
        )

    # Run sample inference
    sample = np.zeros((1, FEATURE_DIM), dtype=np.float32)
    results = session.run(["probabilities"], {"input": sample})
    probs = results[0]

    if probs.shape != (1, OUTPUT_DIM):
        raise ValueError(
            f"Expected output shape (1, {OUTPUT_DIM}), got {probs.shape}"
        )
    if probs.dtype != np.float32:
        raise ValueError(f"Expected float32 output, got {probs.dtype}")

    print(f"  Validation passed: input=({inputs[0].shape}), "
          f"output=({probs.shape}), dtype={probs.dtype}", file=sys.stderr)


def export_to_onnx(model_path: str, onnx_path: str) -> None:
    """Export a joblib model to ONNX format matching the MLScorer contract.

    Args:
        model_path: Path to the joblib model file.
        onnx_path: Path for the output ONNX file.
    """
    print(f"Loading model from {model_path}...", file=sys.stderr)
    model = joblib.load(model_path)

    print("Converting to ONNX...", file=sys.stderr)
    if _is_xgboost_model(model):
        onnx_model = _convert_xgboost_to_onnx(model)
    elif _is_sklearn_model(model):
        onnx_model = _convert_sklearn_to_onnx(model)
    else:
        raise ValueError(
            f"Unsupported model type: {type(model).__name__}. "
            "Expected LogisticRegression, RandomForestClassifier, or XGBClassifier."
        )

    # Rename outputs to match contract
    onnx_model = _rename_outputs(onnx_model)

    # Save
    output_path = Path(onnx_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(onnx_model, str(output_path))
    print(f"Saved ONNX model to {onnx_path}", file=sys.stderr)

    # Validate
    print("Validating ONNX model...", file=sys.stderr)
    _validate_onnx(onnx_path)


def main() -> None:
    """Main entry point for the ONNX export script."""
    parser = argparse.ArgumentParser(
        description="Export a trained model to ONNX format for the MLScorer"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained model (joblib format)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the output ONNX model",
    )
    args = parser.parse_args()

    export_to_onnx(args.model, args.output)
    print("\nExport complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
