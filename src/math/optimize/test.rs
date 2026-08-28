use super::{
    super::{Erase, Scalar, Tensor, assert::AssertionError, sparse::SparseError, special},
    OptimizationError,
};

pub fn rosenbrock<T>(x: &T) -> Result<Scalar, String>
where
    T: Tensor,
    T::Item: Copy + Erase<Erased = Scalar>,
{
    Ok(special::rosenbrock(x, 1.0, 100.0))
}

pub fn rosenbrock_derivative<T>(x: &T) -> Result<T, String>
where
    T: FromIterator<Scalar> + Tensor,
    T::Item: Copy + Erase<Erased = Scalar>,
{
    Ok(special::rosenbrock_derivative(x, 1.0, 100.0))
}

// pub fn rosenbrock_second_derivative<T, U>(x: &T) -> Result<U, String>
// where
//     T: FromIterator<Scalar> + Tensor<Item = Scalar>,
//     U: Tensor<Item = T>,
// {
//     Ok(special::rosenbrock_second_derivative(x, 1.0, 100.0))
// }

#[test]
fn debug() {
    let _ = format!(
        "{:?}",
        OptimizationError::MaximumStepsReached(1, "foo".to_string())
    );
    let _ = format!(
        "{:?}",
        OptimizationError::NotMinimum("foo".to_string(), "bar".to_string())
    );
}

#[test]
fn display() {
    let _ = format!(
        "{}",
        OptimizationError::MaximumStepsReached(1, "foo".to_string())
    );
    let _ = format!(
        "{}",
        OptimizationError::NotMinimum("foo".to_string(), "bar".to_string())
    );
}

#[test]
fn into_test_error() {
    let optimize_error = OptimizationError::MaximumStepsReached(1, "foo".to_string());
    let _: AssertionError = optimize_error.into();
}

#[test]
fn sparse_errors_keep_their_kind() {
    assert!(matches!(
        OptimizationError::from(SparseError::Singular),
        OptimizationError::SingularMatrix
    ));
    assert!(matches!(
        OptimizationError::from(SparseError::Unsymmetric),
        OptimizationError::UnsymmetricMatrix
    ))
}
