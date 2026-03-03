//! Feature vector to tensor conversion for ONNX model input.

use comment_lint_core::features::FeatureVector;

/// The number of f32 elements produced by [`feature_vector_to_tensor`].
///
/// This equals the number of numeric features extracted from a [`FeatureVector`],
/// excluding string fields like `adjacent_node_kind`.
pub const TENSOR_DIM: usize = 16;

/// Convert a [`FeatureVector`] into a flat `Vec<f32>` suitable for ONNX model input.
///
/// The mapping is:
///
/// | Index | Field                          | Encoding           |
/// |-------|--------------------------------|--------------------|
/// | 0     | `token_overlap_jaccard`        | direct f32         |
/// | 1     | `identifier_substring_ratio`   | direct f32         |
/// | 2     | `comment_token_count`          | usize as f32       |
/// | 3     | `is_doc_comment`               | bool → 0.0 / 1.0   |
/// | 4     | `is_before_declaration`        | bool → 0.0 / 1.0   |
/// | 5     | `is_inline`                    | bool → 0.0 / 1.0   |
/// | 6     | `nesting_depth`                | usize as f32       |
/// | 7     | `has_why_indicator`            | bool → 0.0 / 1.0   |
/// | 8     | `has_external_ref`             | bool → 0.0 / 1.0   |
/// | 9     | `imperative_verb_noun`         | bool → 0.0 / 1.0   |
/// | 10    | `is_section_label`             | bool → 0.0 / 1.0   |
/// | 11    | `contains_literal_values`      | bool → 0.0 / 1.0   |
/// | 12    | `references_other_files`       | bool → 0.0 / 1.0   |
/// | 13    | `references_specific_functions` | bool → 0.0 / 1.0  |
/// | 14    | `mirrors_data_structure`       | bool → 0.0 / 1.0   |
/// | 15    | `comment_code_age_ratio`       | Option → value or 0.0 |
///
/// The `adjacent_node_kind` string field is **skipped**.
pub fn feature_vector_to_tensor(features: &FeatureVector) -> Vec<f32> {
    let bool_to_f32 = |b: bool| -> f32 {
        if b { 1.0 } else { 0.0 }
    };

    vec![
        features.token_overlap_jaccard,                          // 0
        features.identifier_substring_ratio,                     // 1
        features.comment_token_count as f32,                     // 2
        bool_to_f32(features.is_doc_comment),                    // 3
        bool_to_f32(features.is_before_declaration),             // 4
        bool_to_f32(features.is_inline),                         // 5
        features.nesting_depth as f32,                           // 6
        bool_to_f32(features.has_why_indicator),                 // 7
        bool_to_f32(features.has_external_ref),                  // 8
        bool_to_f32(features.imperative_verb_noun),              // 9
        bool_to_f32(features.is_section_label),                  // 10
        bool_to_f32(features.contains_literal_values),           // 11
        bool_to_f32(features.references_other_files),            // 12
        bool_to_f32(features.references_specific_functions),     // 13
        bool_to_f32(features.mirrors_data_structure),            // 14
        features.comment_code_age_ratio.unwrap_or(0.0),          // 15
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_features() -> FeatureVector {
        FeatureVector {
            token_overlap_jaccard: 0.0,
            identifier_substring_ratio: 0.0,
            comment_token_count: 0,
            is_doc_comment: false,
            is_before_declaration: false,
            is_inline: false,
            adjacent_node_kind: String::new(),
            nesting_depth: 0,
            has_why_indicator: false,
            has_external_ref: false,
            imperative_verb_noun: false,
            verb_noun_matches_identifier: false,
            is_section_label: false,
            contains_literal_values: false,
            references_other_files: false,
            references_specific_functions: false,
            mirrors_data_structure: false,
            comment_code_age_ratio: None,
        }
    }

    #[test]
    fn test_all_zero_features_to_tensor() {
        let fv = zero_features();
        let tensor = feature_vector_to_tensor(&fv);
        assert_eq!(tensor.len(), TENSOR_DIM);
        for (i, &val) in tensor.iter().enumerate() {
            assert_eq!(val, 0.0, "expected 0.0 at index {i}, got {val}");
        }
    }

    #[test]
    fn test_bool_features_encode_as_one() {
        let mut fv = zero_features();
        fv.is_doc_comment = true;
        fv.is_before_declaration = true;
        fv.is_inline = true;
        fv.has_why_indicator = true;
        fv.has_external_ref = true;
        fv.imperative_verb_noun = true;
        fv.is_section_label = true;
        fv.contains_literal_values = true;
        fv.references_other_files = true;
        fv.references_specific_functions = true;
        fv.mirrors_data_structure = true;

        let tensor = feature_vector_to_tensor(&fv);

        // Bool indices: 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14
        let bool_indices = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14];
        for &idx in &bool_indices {
            assert_eq!(tensor[idx], 1.0, "expected 1.0 at index {idx}");
        }
    }

    #[test]
    fn test_float_features_passthrough() {
        let mut fv = zero_features();
        fv.token_overlap_jaccard = 0.75;
        fv.identifier_substring_ratio = 0.3;

        let tensor = feature_vector_to_tensor(&fv);

        assert!((tensor[0] - 0.75).abs() < f32::EPSILON, "index 0: expected 0.75, got {}", tensor[0]);
        assert!((tensor[1] - 0.3).abs() < f32::EPSILON, "index 1: expected 0.3, got {}", tensor[1]);
    }

    #[test]
    fn test_usize_features_convert() {
        let mut fv = zero_features();
        fv.comment_token_count = 42;
        fv.nesting_depth = 3;

        let tensor = feature_vector_to_tensor(&fv);

        assert_eq!(tensor[2], 42.0, "index 2: expected 42.0, got {}", tensor[2]);
        assert_eq!(tensor[6], 3.0, "index 6: expected 3.0, got {}", tensor[6]);
    }

    #[test]
    fn test_option_none_encodes_as_zero() {
        let fv = zero_features(); // comment_code_age_ratio is None
        let tensor = feature_vector_to_tensor(&fv);
        assert_eq!(tensor[15], 0.0, "index 15: expected 0.0 for None, got {}", tensor[15]);
    }

    #[test]
    fn test_option_some_encodes_value() {
        let mut fv = zero_features();
        fv.comment_code_age_ratio = Some(0.8);

        let tensor = feature_vector_to_tensor(&fv);

        assert!((tensor[15] - 0.8).abs() < f32::EPSILON, "index 15: expected 0.8, got {}", tensor[15]);
    }

    #[test]
    fn test_tensor_length_is_16() {
        let fv = zero_features();
        let tensor = feature_vector_to_tensor(&fv);
        assert_eq!(tensor.len(), 16);
    }

    #[test]
    fn test_tensor_dim_constant() {
        assert_eq!(TENSOR_DIM, 16);
    }
}
