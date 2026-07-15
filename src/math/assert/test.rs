use super::{Assert, AssertionError, assert_eq, assert_eq_from_fd, assert_eq_within_tols};
use crate::{
    EPSILON,
    math::{TensorRank1, TensorRank1List},
};

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
fn assert_eq_from_fd_success() -> Result<(), AssertionError> {
    assert_eq_from_fd(
        &TensorRank1::<_, 1>::from([1.0, 2.0, 3.0]),
        &TensorRank1::<_, 1>::from([1.0, 2.0, 3.0]),
    )
}

#[test]
fn assert_eq_from_fd_weak() -> Result<(), AssertionError> {
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

#[test]
#[should_panic(expected = "Assertion `left == right` failed.")]
fn assert_eq_fail_new() {
    Assert::eq(&0.0, &1.0).unwrap()
}

#[test]
#[should_panic(expected = "Assertion `left ≈= right` failed in 2 places.")]
fn assert_eq_within_tols_fail_new() {
    Assert::default()
        .eq_within_tols(
            &TensorRank1::<_, 1>::from([1.0, 2.0, 3.0]),
            &TensorRank1::<_, 1>::from([3.0, 2.0, 1.0]),
        )
        .unwrap()
}

#[test]
#[should_panic(expected = "Assertion `left ≈= right` failed in 2 places.")]
fn assert_eq_within_fd_tol_fail_new() {
    Assert::default()
        .eq_within_fd_tol(
            &TensorRank1::<_, 1>::from([1.0, 2.0, 3.0]),
            &TensorRank1::<_, 1>::from([3.0, 2.0, 1.0]),
        )
        .unwrap()
}

#[test]
fn assert_eq_within_fd_tol_success_new() {
    Assert::default()
        .eq_within_fd_tol(
            &TensorRank1::<_, 1>::from([1.0, 2.0, 3.0]),
            &TensorRank1::<_, 1>::from([1.0, 2.0, 3.0]),
        )
        .unwrap()
}
