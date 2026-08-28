use super::{Style, defeat_message_inner};

#[test]
fn defeat_message_inner_covers_all_branches() {
    for n in 0..=14 {
        defeat_message_inner(n);
    }
}

#[test]
fn detect_inner_covers_all_branches() {
    for no_color_unset in [false, true] {
        for is_terminal in [false, true] {
            Style::detect_inner(no_color_unset, is_terminal);
        }
    }
}

#[test]
fn error_types_propagate_into_a_boxed_dyn_error() -> Result<(), Box<dyn std::error::Error>> {
    use crate::math::{TensorError, sparse::SparseError};
    fn fails() -> Result<(), TensorError> {
        Err(TensorError::NotPositiveDefinite)
    }
    fn boxes() -> Result<(), Box<dyn std::error::Error>> {
        fails()?;
        Err(Box::new(SparseError::Unsymmetric))
    }
    assert!(boxes().is_err());
    Ok(())
}
