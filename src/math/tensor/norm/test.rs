use super::Norm;
use crate::math::{Tensor, TensorRank1};

fn v() -> TensorRank1<4, 1> {
    TensorRank1::from([1.0, 2.0, 3.0, 4.0])
}

#[test]
fn chebyshev() {
    assert_eq!(Norm::Chebyshev.apply(&v()), 4.0);
}

#[test]
fn euclidean() {
    assert_eq!(Norm::Euclidean.apply(&v()), 30_f64.sqrt());
}

#[test]
fn manhattan() {
    assert_eq!(Norm::Manhattan.apply(&v()), 10.0);
}

#[test]
fn minkowski() {
    assert_eq!(Norm::Minkowski(3.0).apply(&v()), 100_f64.powf(1.0 / 3.0));
}

#[test]
fn norm_p_sum() {
    assert_eq!(v().norm_p_sum(3.0), 100.0);
}

#[test]
fn default_is_euclidean() {
    assert_eq!(Norm::default().apply(&v()), 30_f64.sqrt());
}
