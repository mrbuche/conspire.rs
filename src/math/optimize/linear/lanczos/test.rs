use super::{Lanczos, LanczosError};
use crate::math::{Scalar, SquareMatrix, Vector};
use std::f64::consts::PI;

fn stiffness(n: usize) -> SquareMatrix {
    let mut k = SquareMatrix::zero(n);
    for i in 0..n {
        k[i][i] = if i + 1 == n { 1.0 } else { 2.0 };
        if i + 1 < n {
            k[i][i + 1] = -1.0;
            k[i + 1][i] = -1.0;
        }
    }
    k
}

fn analytic(n: usize) -> Vec<Scalar> {
    (1..=n)
        .map(|k| 2.0 - 2.0 * ((2 * k - 1) as Scalar * PI / (2 * n + 1) as Scalar).cos())
        .collect()
}

fn masses(n: usize) -> Vec<Scalar> {
    (0..n).map(|i| 1.0 + 0.5 * (i % 3) as Scalar).collect()
}

fn diagonal(m: &[Scalar]) -> impl FnMut(&Vector) -> Vector + '_ {
    move |x| x.as_slice().iter().zip(m).map(|(x, m)| x * m).collect()
}

fn shifted(k: &SquareMatrix, m: &[Scalar], shift: Scalar) -> SquareMatrix {
    let mut a = k.clone();
    m.iter().enumerate().for_each(|(i, m)| a[i][i] -= shift * m);
    a
}

fn inverse<'a>(a: &'a SquareMatrix) -> impl FnMut(&Vector) -> Vector + 'a {
    |x| a.clone().solve_lu(x).unwrap()
}

fn residual(k: &SquareMatrix, m: &[Scalar], value: Scalar, x: &Vector) -> Scalar {
    let kx = k.clone() * x;
    let mx: Vec<Scalar> = x.as_slice().iter().zip(m).map(|(x, m)| x * m).collect();
    kx.as_slice()
        .iter()
        .zip(&mx)
        .map(|(kx, mx)| (kx - value * mx).abs())
        .fold(0.0, Scalar::max)
}

fn mass_dot(m: &[Scalar], a: &Vector, b: &Vector) -> Scalar {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .zip(m)
        .map(|((a, b), m)| a * b * m)
        .sum()
}

#[test]
fn lowest_modes_of_a_fixed_free_chain() {
    let n = 40;
    let k = stiffness(n);
    let m = vec![1.0; n];
    let a = shifted(&k, &m, -0.1);
    let pairs = Lanczos::default()
        .solve(inverse(&a), diagonal(&m), -0.1, n, 4)
        .unwrap();
    analytic(n)
        .iter()
        .zip(&pairs.values)
        .for_each(|(expected, value)| assert!((expected - value).abs() < 1e-10));
    assert_eq!(pairs.values.len(), 4);
}

#[test]
fn nonuniform_mass_residuals_and_orthonormality() {
    let n = 30;
    let k = stiffness(n);
    let m = masses(n);
    let a = shifted(&k, &m, -0.05);
    let pairs = Lanczos::default()
        .solve(inverse(&a), diagonal(&m), -0.05, n, 5)
        .unwrap();
    for (i, (value, x)) in pairs.values.iter().zip(&pairs.vectors).enumerate() {
        assert!(residual(&k, &m, *value, x) < 1e-8);
        for (j, y) in pairs.vectors.iter().enumerate() {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((mass_dot(&m, x, y) - expected).abs() < 1e-9);
        }
    }
    assert!(pairs.values.windows(2).all(|w| w[0] < w[1]));
}

#[test]
fn modes_nearest_an_interior_shift() {
    let n = 40;
    let k = stiffness(n);
    let m = vec![1.0; n];
    let shift = 2.0;
    let a = shifted(&k, &m, shift);
    let pairs = Lanczos::default()
        .solve(inverse(&a), diagonal(&m), shift, n, 3)
        .unwrap();
    let mut expected = analytic(n);
    expected.sort_by(|a, b| (a - shift).abs().total_cmp(&(b - shift).abs()));
    let mut expected = expected[..3].to_vec();
    expected.sort_by(Scalar::total_cmp);
    expected
        .iter()
        .zip(&pairs.values)
        .for_each(|(expected, value)| assert!((expected - value).abs() < 1e-9));
}

#[test]
fn the_whole_spectrum_is_recovered_exactly() {
    let n = 12;
    let k = stiffness(n);
    let m = vec![1.0; n];
    let a = shifted(&k, &m, -0.1);
    let pairs = Lanczos::default()
        .solve(inverse(&a), diagonal(&m), -0.1, n, n)
        .unwrap();
    analytic(n)
        .iter()
        .zip(&pairs.values)
        .for_each(|(expected, value)| assert!((expected - value).abs() < 1e-10));
}

#[test]
fn rejects_an_impossible_count() {
    let n = 5;
    let k = stiffness(n);
    let m = vec![1.0; n];
    let a = shifted(&k, &m, -0.1);
    let error = Lanczos::default()
        .solve(inverse(&a), diagonal(&m), -0.1, n, n + 1)
        .unwrap_err();
    assert!(matches!(error, LanczosError::InvalidCount(6, 5)));
}

#[test]
fn reports_an_unconverged_solve() {
    let n = 40;
    let k = stiffness(n);
    let m = vec![1.0; n];
    let a = shifted(&k, &m, -0.1);
    let error = Lanczos {
        max_steps: 4,
        ..Default::default()
    }
    .solve(inverse(&a), diagonal(&m), -0.1, n, 4)
    .unwrap_err();
    assert!(matches!(error, LanczosError::MaximumStepsReached(4)));
}
