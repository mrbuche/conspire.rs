use super::{Scalar, Tensor, TensorError};
use crate::{ABS_TOL, REL_TOL, defeat_message};
use std::{
    cmp::PartialEq,
    fmt::{self, Debug, Display, Formatter},
};

#[cfg(test)]
use crate::EPSILON;

#[cfg(test)]
use super::rank_1::{TensorRank1, list::TensorRank1List};

#[cfg(test)]
pub trait ErrorTensor {
    fn error_fd(&self, comparator: &Self, epsilon: Scalar) -> Option<(bool, usize)>;
}

pub fn assert_eq<'a, T>(value_1: &'a T, value_2: &'a T) -> Result<(), TestError>
where
    T: Display + PartialEq,
{
    if value_1 == value_2 {
        Ok(())
    } else {
        Err(TestError {
            message: format!(
                "\n\x1b[1;91mAssertion `left == right` failed.\n\x1b[0;91m  left: {value_1}\n right: {value_2}\x1b[0m"
            ),
        })
    }
}

#[cfg(test)]
pub fn assert_eq_from_fd<'a, T>(value: &'a T, value_fd: &'a T) -> Result<(), TestError>
where
    T: Display + ErrorTensor + Tensor,
{
    if let Some((failed, count)) = value.error_fd(value_fd, 3.0 * EPSILON) {
        if failed {
            let abs = value.sub_abs(value_fd);
            let rel = value.sub_rel(value_fd);
            Err(TestError {
                message: format!(
                    "\n\x1b[1;91mAssertion `left ≈= right` failed in {count} places.\n\x1b[0;91m  left: {value}\n right: {value_fd}\n   abs: {abs}\n   rel: {rel}\x1b[0m"
                ),
            })
        } else {
            println!(
                "Warning: \n\x1b[1;93mAssertion `left ≈= right` was weak in {count} places.\x1b[0m"
            );
            Ok(())
        }
    } else {
        Ok(())
    }
}

pub fn assert_eq_within<'a, T>(
    value_1: &'a T,
    value_2: &'a T,
    tol_abs: Scalar,
    tol_rel: Scalar,
) -> Result<(), TestError>
where
    T: Display + Tensor,
{
    if let Some(count) = value_1.error_count(value_2, tol_abs, tol_rel) {
        let abs = value_1.sub_abs(value_2);
        let rel = value_1.sub_rel(value_2);
        Err(TestError {
            message: format!(
                "\n\x1b[1;91mAssertion `left ≈= right` failed in {count} places.\n\x1b[0;91m  left: {value_1}\n right: {value_2}\n   abs: {abs}\n   rel: {rel}\x1b[0m"
            ),
        })
    } else {
        Ok(())
    }
}

pub fn assert_eq_within_tols<'a, T>(value_1: &'a T, value_2: &'a T) -> Result<(), TestError>
where
    T: Display + Tensor,
{
    assert_eq_within(value_1, value_2, ABS_TOL, REL_TOL)
}

pub struct TestError {
    pub message: String,
}

impl Debug for TestError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}\n\x1b[0;2;31m{}\x1b[0m\n",
            self.message,
            defeat_message()
        )
    }
}

impl From<String> for TestError {
    fn from(error: String) -> Self {
        Self { message: error }
    }
}

impl From<&str> for TestError {
    fn from(error: &str) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl From<TensorError> for TestError {
    fn from(error: TensorError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

#[test]
fn test_error_from_string() {
    assert_eq!(
        TestError::from("An error occurred".to_string()).message,
        "An error occurred"
    );
}

#[test]
fn test_error_from_str() {
    assert_eq!(
        TestError::from("An error occurred").message,
        "An error occurred"
    );
}

#[test]
fn test_error_from_tensor_error() {
    let tensor_error = TensorError::NotPositiveDefinite;
    let _ = format!("{:?}", tensor_error);
    let _ = TestError::from(tensor_error);
}

#[test]
#[should_panic(expected = "Assertion `left == right` failed.")]
fn assert_eq_fail() {
    assert_eq(&0.0, &1.0).unwrap()
}

#[test]
#[should_panic(expected = "Assertion `left ≈= right` failed in 2 places.")]
fn assert_eq_from_fd_fail() {
    assert_eq_from_fd(
        &TensorRank1::<_, 1>::from([1.0, 2.0, 3.0]),
        &TensorRank1::<_, 1>::from([3.0, 2.0, 1.0]),
    )
    .unwrap()
}

#[test]
fn assert_eq_from_fd_success() -> Result<(), TestError> {
    assert_eq_from_fd(
        &TensorRank1::<_, 1>::from([1.0, 2.0, 3.0]),
        &TensorRank1::<_, 1>::from([1.0, 2.0, 3.0]),
    )
}

#[test]
fn assert_eq_from_fd_weak() -> Result<(), TestError> {
    assert_eq_from_fd(
        &TensorRank1List::<_, 1, 1>::from([[EPSILON * 1.01]]),
        &TensorRank1List::<_, 1, 1>::from([[EPSILON * 1.02]]),
    )
}

#[test]
#[should_panic(expected = "Assertion `left ≈= right` failed in 2 places.")]
fn assert_eq_within_tols_fail() {
    assert_eq_within_tols(
        &TensorRank1::<_, 1>::from([1.0, 2.0, 3.0]),
        &TensorRank1::<_, 1>::from([3.0, 2.0, 1.0]),
    )
    .unwrap()
}
