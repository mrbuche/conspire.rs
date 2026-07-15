use super::CscMatrix;
use crate::math::{
    Vector,
    assert::{AssertionError, assert_eq_within_tols},
};

fn laplacian(k: usize) -> CscMatrix {
    let n = k * k;
    let mut pattern: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
    (0..k).for_each(|a| {
        (0..k).for_each(|b| {
            let i = k * a + b;
            if b + 1 < k {
                pattern.push((i, i + 1));
                pattern.push((i + 1, i));
            }
            if a + 1 < k {
                pattern.push((i, i + k));
                pattern.push((i + k, i));
            }
        })
    });
    let mut matrix = CscMatrix::from_pattern(n, n, pattern);
    matrix.fill(|i, j| if i == j { 4.0 } else { -1.0 });
    matrix
}

fn asymmetric(n: usize) -> CscMatrix {
    let mut pattern: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
    (0..n).for_each(|i| {
        pattern.push((i, (3 * i + 1) % n));
        pattern.push(((5 * i + 2) % n, i));
        pattern.push((i, (i + 7) % n));
    });
    let mut matrix = CscMatrix::from_pattern(n, n, pattern);
    matrix.fill(|i, j| if i == j { 8.0 } else { -1.0 });
    matrix
}

fn assert_permutation(permutation: &[usize], n: usize) {
    assert_eq!(permutation.len(), n);
    let mut sorted = permutation.to_vec();
    sorted.sort_unstable();
    assert!(sorted.into_iter().eq(0..n));
}

#[test]
fn permutation_asymmetric() {
    assert_permutation(&asymmetric(100).amd(), 100);
}

#[test]
fn permutation_diagonal() {
    let mut matrix = CscMatrix::from_pattern(5, 5, (0..5).map(|i| (i, i)).collect());
    matrix.fill(|_, _| 1.0);
    assert_permutation(&matrix.amd(), 5);
}

#[test]
fn permutation_laplacian() {
    assert_permutation(&laplacian(20).amd(), 400);
}

#[test]
fn reduces_fill() {
    let matrix = laplacian(20);
    let natural = matrix.lu().expect("Factorization failed.").nonzeros();
    let ordered = matrix.lu_amd().expect("Factorization failed.").nonzeros();
    assert!(2 * ordered < natural);
}

#[test]
fn solve_laplacian() -> Result<(), AssertionError> {
    let matrix = laplacian(20);
    let b: Vector = (0..400).map(|i| (i % 17) as f64 - 8.0).collect();
    let lu = matrix.lu_amd().map_err(|_| AssertionError {
        message: "Factorization failed.".to_string(),
    })?;
    assert_eq_within_tols(&(&matrix * &lu.solve(&b)), &b)
}

#[test]
fn solve_asymmetric() -> Result<(), AssertionError> {
    let matrix = asymmetric(100);
    let b: Vector = (0..100).map(|i| (i % 13) as f64 - 6.0).collect();
    let lu = matrix.lu_amd().map_err(|_| AssertionError {
        message: "Factorization failed.".to_string(),
    })?;
    assert_eq_within_tols(&(&matrix * &lu.solve(&b)), &b)
}
