use super::{CscMatrix, SparseError, Vector};
use crate::math::assert::{AssertionError, assert_eq_within_tols};

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
fn solve_matches_lu() -> Result<(), AssertionError> {
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
fn refactor_new_values() -> Result<(), AssertionError> {
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
fn indefinite() -> Result<(), AssertionError> {
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
fn missing_diagonal() -> Result<(), AssertionError> {
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
    let mut matrix = CscMatrix::from_pattern(3, 3, pattern);
    matrix.fill(|i, j| if i == j { 4.0 } else { (i + j) as f64 });
    let mut ldl = matrix.ldl_symbolic()?;
    ldl.refactor(&matrix).expect("Refactorization failed.");
    let b = Vector::from([1.0, 2.0, 3.0]);
    assert_eq_within_tols(&(&matrix * &ldl.solve(&b)), &b)
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

fn saddle_point_matrix(n: usize, num_constraints: usize, scale: f64) -> CscMatrix {
    let total = n + num_constraints;
    let mut pattern: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
    (0..n - 3).for_each(|i| {
        pattern.push((i, i + 3));
        pattern.push((i + 3, i));
    });
    (0..num_constraints).for_each(|c| {
        let dof = (5 * c) % n;
        let other = (5 * c + 2) % n;
        pattern.push((n + c, dof));
        pattern.push((dof, n + c));
        pattern.push((n + c, other));
        pattern.push((other, n + c));
    });
    let mut matrix = CscMatrix::from_pattern(total, total, pattern);
    matrix.fill(|i, j| {
        if i == j {
            16.0 + (i % 5) as f64
        } else if i >= n || j >= n {
            1.0 + (i.min(j) % 3) as f64
        } else {
            scale * (((i.min(j) * 7 + i.max(j) * 3) % 5) as f64 - 2.0)
        }
    });
    matrix
}

#[test]
fn saddle_point_matches_lu() -> Result<(), AssertionError> {
    let matrix = saddle_point_matrix(40, 6, 1.0);
    let b: Vector = (0..46).map(|i| (i % 13) as f64 - 6.0).collect();
    let mut ldl = matrix.ldl_symbolic()?;
    ldl.refactor(&matrix).expect("Refactorization failed.");
    assert_eq_within_tols(
        &ldl.solve(&b),
        &matrix.lu_amd().expect("Factorization failed.").solve(&b),
    )?;
    assert_eq_within_tols(&(&matrix * &ldl.solve(&b)), &b)
}

#[test]
fn saddle_point_refactor_new_values() -> Result<(), AssertionError> {
    let matrix = saddle_point_matrix(40, 6, 1.0);
    let mut ldl = matrix.ldl_symbolic()?;
    ldl.refactor(&matrix).expect("Refactorization failed.");
    let matrix = saddle_point_matrix(40, 6, -2.0);
    ldl.refactor(&matrix).expect("Refactorization failed.");
    let b: Vector = (0..46).map(|i| (i % 17) as f64 - 8.0).collect();
    assert_eq_within_tols(&(&matrix * &ldl.solve(&b)), &b)
}

#[test]
fn saddle_point_larger() -> Result<(), AssertionError> {
    let matrix = saddle_point_matrix(200, 31, 1.0);
    let b: Vector = (0..231).map(|i| (i % 11) as f64 - 5.0).collect();
    let mut ldl = matrix.ldl_symbolic()?;
    ldl.refactor(&matrix).expect("Refactorization failed.");
    assert_eq_within_tols(&(&matrix * &ldl.solve(&b)), &b)
}
