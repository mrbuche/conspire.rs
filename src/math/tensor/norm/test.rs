use super::Norm;
use crate::math::Current;
use crate::math::{Tensor, TensorRank1, Vector, random::random_normal};

fn v() -> TensorRank1<4, Current> {
    TensorRank1::from([1.0, 2.0, 3.0, 4.0])
}

fn kahan_dot(a: &[f64], b: &[f64]) -> f64 {
    let (mut sum, mut c) = (0.0, 0.0);
    a.iter().zip(b).for_each(|(&x, &y)| {
        let term = x * y - c;
        let next = sum + term;
        c = (next - sum) - term;
        sum = next;
    });
    sum
}

#[test]
fn full_contraction_matches_compensated_reference() {
    let data: Vec<f64> = (0..30_000).map(|_| random_normal(0.0, 1.0)).collect();
    let vector = Vector::from(data.clone());
    let reference = kahan_dot(&data, &data);
    let contraction = vector.full_contraction(&vector);
    assert!(
        (contraction / reference - 1.0).abs() < 1e-9,
        "{contraction} vs {reference}"
    );
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
