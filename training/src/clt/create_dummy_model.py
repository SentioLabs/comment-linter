"""Create a minimal ONNX model for testing the MLScorer.

This generates a simple linear model that takes 16 float features
and outputs two values: [P(not_superfluous), P(superfluous)].

The model computes:
  logit = sum(features * weights) + bias
  P(superfluous) = sigmoid(logit)
  P(not_superfluous) = 1 - P(superfluous)

Weights are set so that positive heuristic signals (indices 0,1,3-5,9-14)
increase the superfluous probability, while negative signals (7,8)
decrease it. This mimics the heuristic scorer's behavior for testing.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper

from clt.utils import FEATURE_DIM, OUTPUT_DIM


def create_dummy_model() -> onnx.ModelProto:
    # Weights: positive for superfluous signals, negative for valuable signals
    weights = np.array(
        [
            0.5,   # 0: token_overlap_jaccard (superfluous)
            0.4,   # 1: identifier_substring_ratio (superfluous)
            0.0,   # 2: comment_token_count (neutral)
            0.1,   # 3: is_doc_comment (slightly superfluous for testing)
            0.1,   # 4: is_before_declaration
            0.0,   # 5: is_inline (neutral)
            0.0,   # 6: nesting_depth (neutral)
            -0.3,  # 7: has_why_indicator (valuable)
            -0.2,  # 8: has_external_ref (valuable)
            0.3,   # 9: imperative_verb_noun (superfluous)
            0.2,   # 10: is_section_label (superfluous)
            0.1,   # 11: contains_literal_values
            0.1,   # 12: references_other_files
            0.1,   # 13: references_specific_functions
            0.1,   # 14: mirrors_data_structure
            0.0,   # 15: comment_code_age_ratio (neutral)
        ],
        dtype=np.float32,
    )

    bias = np.array([-0.5], dtype=np.float32)

    # --- Build the ONNX graph ---
    # Input: [batch, 16]
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, FEATURE_DIM])
    # Output: [batch, 2] probabilities
    Y = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, OUTPUT_DIM])

    # Initializers
    W_init = helper.make_tensor("W", TensorProto.FLOAT, [FEATURE_DIM, 1], weights.flatten())
    B_init = helper.make_tensor("B", TensorProto.FLOAT, [1], bias.flatten())
    one_init = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])

    # logit = X @ W + B
    matmul = helper.make_node("MatMul", ["input", "W"], ["matmul_out"])
    add_bias = helper.make_node("Add", ["matmul_out", "B"], ["logit"])

    # p_super = sigmoid(logit)
    sigmoid = helper.make_node("Sigmoid", ["logit"], ["p_super"])

    # p_not = 1 - p_super
    sub = helper.make_node("Sub", ["one", "p_super"], ["p_not"])

    # probabilities = concat([p_not, p_super], axis=1)
    concat = helper.make_node("Concat", ["p_not", "p_super"], ["probabilities"], axis=1)

    graph = helper.make_graph(
        [matmul, add_bias, sigmoid, sub, concat],
        "dummy_classifier",
        [X],
        [Y],
        initializer=[W_init, B_init, one_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main():
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="Create a dummy ONNX model for testing")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("crates/comment-lint-ml/tests/fixtures/dummy_model.onnx"),
        help="Output path for the ONNX model",
    )
    args = parser.parse_args()

    model = create_dummy_model()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"Saved dummy model to {args.output}")
    print(f"  Input: 'input' shape=[batch, {FEATURE_DIM}] float32")
    print(f"  Output: 'probabilities' shape=[batch, {OUTPUT_DIM}] float32")


if __name__ == "__main__":
    main()
