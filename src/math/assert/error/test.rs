use super::AssertionError;
use crate::math::TensorError;

#[test]
fn test_error_from_string() {
    assert_eq!(
        AssertionError::from("An error occurred".to_string()).message,
        "An error occurred"
    );
}

#[test]
fn test_error_from_str() {
    assert_eq!(
        AssertionError::from("An error occurred").message,
        "An error occurred"
    );
}

#[test]
fn test_error_from_tensor_error() {
    let tensor_error = TensorError::NotPositiveDefinite;
    let _ = format!("{:?}", tensor_error);
    let _ = AssertionError::from(tensor_error);
}
