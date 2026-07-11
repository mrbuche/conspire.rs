use super::{CscMatrix, SparseError, Vector};
use crate::math::test::{TestError, assert_eq_within_tols};

fn symmetric_matrix(n: usize, scale: f64) -> CscMatrix {
    let mut pattern: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
    (0..n - 7).for_each(|i| {
        pattern.push((i, i + 7));
        pattern.push((i + 7, i));
    });
    (0..n - 3).for_each(|i| {
        pattern.push((i, i + 3));
        pattern.push((i + 3, i));
    });
    let mut matrix = CscMatrix::from_pattern(n, n, pattern);
    matrix.fill(|i, j| {
        if i == j {
            16.0 + (i % 5) as f64
        } else {
            scale * (((i.min(j) * 7 + i.max(j) * 3) % 5) as f64 - 2.0)
        }
    });
    matrix
}

#[test]
fn solve_matches_lu() -> Result<(), TestError> {
    let n = 100;
    let matrix = symmetric_matrix(n, 1.0);
    let b: Vector = (0..n).map(|i| (i % 13) as f64 - 6.0).collect();
    let mut ldl = matrix.ldl_symbolic()?;
    ldl.refactor(&matrix).expect("Refactorization failed.");
    assert_eq_within_tols(
        &ldl.solve(&b),
        &matrix.lu_amd().expect("Factorization failed.").solve(&b),
    )?;
    assert_eq_within_tols(&(&matrix * &ldl.solve(&b)), &b)
}

#[test]
fn refactor_new_values() -> Result<(), TestError> {
    let n = 100;
    let matrix = symmetric_matrix(n, 1.0);
    let mut ldl = matrix.ldl_symbolic()?;
    ldl.refactor(&matrix).expect("Refactorization failed.");
    let matrix = symmetric_matrix(n, -2.0);
    ldl.refactor(&matrix).expect("Refactorization failed.");
    let b: Vector = (0..n).map(|i| (i % 17) as f64 - 8.0).collect();
    assert_eq_within_tols(&(&matrix * &ldl.solve(&b)), &b)
}

#[test]
fn indefinite() -> Result<(), TestError> {
    let n = 100;
    let mut matrix = symmetric_matrix(n, 1.0);
    matrix.fill(|i, j| {
        if i == j {
            (16.0 + (i % 5) as f64) * if i % 2 == 0 { 1.0 } else { -1.0 }
        } else {
            ((i.min(j) * 7 + i.max(j) * 3) % 5) as f64 - 2.0
        }
    });
    let mut ldl = matrix.ldl_symbolic()?;
    ldl.refactor(&matrix).expect("Refactorization failed.");
    let b: Vector = (0..n).map(|i| (i % 13) as f64 - 6.0).collect();
    assert_eq_within_tols(&(&matrix * &ldl.solve(&b)), &b)
}

#[test]
fn missing_diagonal() {
    let pattern = vec![
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 2),
        (2, 0),
        (1, 2),
        (2, 1),
    ];
    let matrix = CscMatrix::from_pattern(3, 3, pattern);
    assert!(
        matrix
            .ldl_symbolic()
            .is_err_and(|error| error == SparseError::Unsymmetric)
    )
}

#[test]
fn singular() {
    let mut matrix = CscMatrix::from_pattern(2, 2, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    matrix.fill(|_, _| 1.0);
    let mut ldl = matrix.ldl_symbolic().expect("Symbolic failed.");
    assert!(
        ldl.refactor(&matrix)
            .is_err_and(|error| error == SparseError::Singular)
    )
}
