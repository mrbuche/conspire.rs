use super::{
    super::{Scalar, Tensor, special, test::TestError},
    OptimizationError,
};

pub fn rosenbrock<T>(x: &T) -> Result<Scalar, OptimizationError>
where
    T: Tensor<Item = Scalar>,
{
    Ok(special::rosenbrock(x, 1.0, 100.0))
}

pub fn rosenbrock_derivative<T>(x: &T) -> Result<T, OptimizationError>
where
    T: FromIterator<Scalar> + Tensor<Item = Scalar>,
{
    Ok(special::rosenbrock_derivative(x, 1.0, 100.0))
}

// pub fn rosenbrock_second_derivative<T, U>(x: &T) -> Result<U, OptimizationError>
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
    let _: TestError = optimize_error.into();
}
