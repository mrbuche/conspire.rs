use super::{exponents, monomials, moving_least_squares, quartic_weight};
use crate::math::{SquareMatrixError, random::Rng};
use std::array::from_fn;

const TOLERANCE: f64 = 1e-10;

fn cloud<const D: usize>(seed: u64, count: usize) -> (Vec<[f64; D]>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let centers = (0..count)
        .map(|_| from_fn(|_| rng.uniform() - 0.5))
        .collect();
    let weights = (0..count).map(|_| 0.1 + 0.9 * rng.uniform()).collect();
    (centers, weights)
}

fn reproduces<const D: usize>(degree: usize, count: usize) {
    let (centers, weights) = cloud::<D>(1, count);
    let point: [f64; D] = from_fn(|k| 0.05 * (k as f64 + 1.0));
    let psi = moving_least_squares(point, &centers, &weights, degree).unwrap();
    let exponents = exponents::<D>(degree);
    let target = monomials(&exponents, &point);
    (0..exponents.len()).for_each(|m| {
        let sum: f64 = centers
            .iter()
            .zip(&psi)
            .map(|(center, p)| p * monomials(&exponents, center)[m])
            .sum();
        assert!(
            (sum - target[m]).abs() < TOLERANCE,
            "degree {degree}, monomial {m}: {sum} vs {}",
            target[m]
        );
    });
}

#[test]
fn reproduces_polynomials_2d() {
    reproduces::<2>(0, 6);
    reproduces::<2>(1, 8);
    reproduces::<2>(2, 14);
    reproduces::<2>(3, 20);
}

#[test]
fn reproduces_polynomials_3d() {
    reproduces::<3>(1, 10);
    reproduces::<3>(2, 24);
}

#[test]
fn counts_monomials() {
    assert_eq!(exponents::<2>(1).len(), 3);
    assert_eq!(exponents::<2>(2).len(), 6);
    assert_eq!(exponents::<3>(1).len(), 4);
    assert_eq!(exponents::<3>(2).len(), 10);
    assert_eq!(exponents::<2>(2)[0], [0, 0]);
}

#[test]
fn degree_zero_is_normalized_weights() {
    let (centers, weights) = cloud::<2>(2, 7);
    let psi = moving_least_squares([0.0, 0.1], &centers, &weights, 0).unwrap();
    let total: f64 = weights.iter().sum();
    psi.iter()
        .zip(&weights)
        .for_each(|(p, w)| assert!((p - w / total).abs() < TOLERANCE));
}

#[test]
fn zero_weight_gives_zero_basis() {
    let (centers, mut weights) = cloud::<2>(3, 9);
    weights[4] = 0.0;
    let psi = moving_least_squares([0.0, 0.0], &centers, &weights, 1).unwrap();
    assert_eq!(psi[4], 0.0);
    assert!((psi.iter().sum::<f64>() - 1.0).abs() < TOLERANCE);
}

#[test]
fn invariant_to_shift_and_scale() {
    let (centers, weights) = cloud::<2>(4, 9);
    let point = [0.02, -0.03];
    let base = moving_least_squares(point, &centers, &weights, 2).unwrap();
    let (scale, shift) = (1.0e3, 5.0e5);
    let moved_centers: Vec<[f64; 2]> = centers
        .iter()
        .map(|c| [c[0] * scale + shift, c[1] * scale - shift])
        .collect();
    let moved = moving_least_squares(
        [point[0] * scale + shift, point[1] * scale - shift],
        &moved_centers,
        &weights,
        2,
    )
    .unwrap();
    base.iter()
        .zip(&moved)
        .for_each(|(a, b)| assert!((a - b).abs() < 1e-8, "{a} vs {b}"));
}

#[test]
fn invariant_to_weight_scale() {
    let (centers, weights) = cloud::<2>(5, 9);
    let base = moving_least_squares([0.0, 0.0], &centers, &weights, 1).unwrap();
    let tiny: Vec<f64> = weights.iter().map(|w| w * 1e-12).collect();
    let scaled = moving_least_squares([0.0, 0.0], &centers, &tiny, 1).unwrap();
    base.iter()
        .zip(&scaled)
        .for_each(|(a, b)| assert!((a - b).abs() < TOLERANCE));
}

#[test]
fn degenerate_supports_are_singular() {
    let weights = [1.0; 4];
    let collinear = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    assert!(matches!(
        moving_least_squares([0.5, 0.5], &collinear, &weights, 1),
        Err(SquareMatrixError::Singular)
    ));
    let few = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    assert!(moving_least_squares([0.2, 0.2], &few, &[1.0; 3], 1).is_ok());
    assert!(moving_least_squares([0.2, 0.2], &few, &[1.0; 3], 2).is_err());
    assert!(matches!(
        moving_least_squares([0.2, 0.2], &few, &[0.0; 3], 1),
        Err(SquareMatrixError::Singular)
    ));
}

#[test]
#[should_panic(expected = "Each center needs a weight.")]
fn mismatched_weights() {
    let _ = moving_least_squares([0.0], &[[0.0], [1.0]], &[1.0], 0);
}

#[test]
fn quartic_weight_shape() {
    assert_eq!(quartic_weight(0.0), 1.0);
    assert_eq!(quartic_weight(1.0), 0.0);
    assert_eq!(quartic_weight(3.0), 0.0);
    assert!((quartic_weight(0.5) - 0.5625).abs() < 1e-15);
    let samples: Vec<f64> = (0..=100)
        .map(|i| quartic_weight(i as f64 / 100.0))
        .collect();
    assert!(samples.windows(2).all(|w| w[1] <= w[0]));
    let slope = (quartic_weight(1.0 - 1e-6) - quartic_weight(1.0)) / 1e-6;
    assert!(slope.abs() < 1e-5, "not flat at the edge: {slope}");
}
