use super::{Tensor, TensorError, TensorRank0};
use crate::{ABS_TOL, REL_TOL, defeat_message};
use std::{
    cmp::PartialEq,
    fmt::{self, Debug, Display, Formatter},
};

#[cfg(test)]
use crate::EPSILON;

#[cfg(test)]
use super::{
    TensorArray,
    rank_1::{TensorRank1, list::TensorRank1List},
};

pub trait ErrorTensor {
    fn error_fd(&self, comparator: &Self, epsilon: &TensorRank0) -> Option<(bool, usize)>;
}

pub fn assert_eq<'a, T>(value_1: &'a T, value_2: &'a T) -> Result<(), TestError>
where
    T: Display + PartialEq + ErrorTensor,
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
    T: Display + ErrorTensor,
{
    if let Some((failed, error_count)) = value.error_fd(value_fd, &EPSILON) {
        if failed {
            Err(TestError {
                message: format!(
                    "\n\x1b[1;91mAssertion `left ≈= right` failed in {error_count} places.\n\x1b[0;91m  left: {value}\n right: {value_fd}\x1b[0m"
                ),
            })
        } else {
            println!(
                "Warning: \n\x1b[1;93mAssertion `left ≈= right` was weak in {error_count} places.\x1b[0m"
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
    tol_abs: &TensorRank0,
    tol_rel: &TensorRank0,
) -> Result<(), TestError>
where
    T: Display + ErrorTensor + Tensor,
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
    T: Display + ErrorTensor + Tensor,
{
    assert_eq_within(value_1, value_2, &ABS_TOL, &REL_TOL)
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

impl From<TensorError> for TestError {
    fn from(error: TensorError) -> TestError {
        Self {
            message: error.to_string(),
        }
    }
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
        &TensorRank1::<3, 1>::new([1.0, 2.0, 3.0]),
        &TensorRank1::<3, 1>::new([3.0, 2.0, 1.0]),
    )
    .unwrap()
}

#[test]
fn assert_eq_from_fd_success() -> Result<(), TestError> {
    assert_eq_from_fd(
        &TensorRank1::<3, 1>::new([1.0, 2.0, 3.0]),
        &TensorRank1::<3, 1>::new([1.0, 2.0, 3.0]),
    )
}

#[test]
fn assert_eq_from_fd_weak() -> Result<(), TestError> {
    assert_eq_from_fd(
        &TensorRank1List::<1, 1, 1>::new([[EPSILON * 1.01]]),
        &TensorRank1List::<1, 1, 1>::new([[EPSILON * 1.02]]),
    )
}

#[test]
#[should_panic(expected = "Assertion `left ≈= right` failed in 2 places.")]
fn assert_eq_within_tols_fail() {
    assert_eq_within_tols(
        &TensorRank1::<3, 1>::new([1.0, 2.0, 3.0]),
        &TensorRank1::<3, 1>::new([3.0, 2.0, 1.0]),
    )
    .unwrap()
}
