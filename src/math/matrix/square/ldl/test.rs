use super::super::{SquareMatrix, Vector};
use crate::math::assert::{Assert, AssertionError};

fn kkt_symmetric_dim_25() -> SquareMatrix {
    let (ng, cg, nl, cl) = (9, 4, 9, 3);
    let (no, n) = (ng + cg, ng + cg + nl + cl);
    let mut matrix = SquareMatrix::zero(n);
    for i in 0..ng {
        for j in 0..ng {
            let entry = (((i + j) * 7) % 11) as f64 - 5.0;
            matrix[i][j] = entry;
            matrix[i][no + j] = entry / 3.0;
            matrix[no + i][j] = entry / 3.0;
            matrix[no + i][no + j] = -entry;
        }
        matrix[i][i] += 9.0;
        matrix[no + i][no + i] -= 9.0;
    }
    for (i, &j) in [0, 1, 2, 5].iter().enumerate() {
        matrix[ng + i][j] = -1.0;
        matrix[j][ng + i] = -1.0;
    }
    for (i, &j) in [1, 2, 5].iter().enumerate() {
        matrix[no + nl + i][no + j] = -1.0;
        matrix[no + j][no + nl + i] = -1.0;
    }
    matrix
}

fn rhs_dim_25() -> Vector {
    (0..25).map(|i| ((i * 5) % 7) as f64 - 3.0).collect()
}

#[test]
fn solve_ldl_kkt_dim_25() -> Result<(), AssertionError> {
    let matrix = kkt_symmetric_dim_25();
    let rhs = rhs_dim_25();
    let solution = matrix.solve_ldl(&rhs).unwrap();
    Assert::default().eq_within_tols(&(matrix * &solution), &rhs)
}

#[test]
fn solve_ldl_matches_lu_dim_25() -> Result<(), AssertionError> {
    let rhs = rhs_dim_25();
    Assert::default().eq_within_tols(
        kkt_symmetric_dim_25().solve_ldl(&rhs).unwrap(),
        &kkt_symmetric_dim_25().solve_lu(&rhs).unwrap(),
    )
}

#[test]
fn solve_ldl_zero_diagonal() -> Result<(), AssertionError> {
    let n = 9;
    let mut matrix = SquareMatrix::zero(n);
    for i in 0..n {
        for j in 0..=i {
            let entry = ((i * 5 + j * 3) % 7) as f64 - 3.0;
            matrix[i][j] = entry;
            matrix[j][i] = entry
        }
    }
    (0..n).step_by(3).for_each(|i| matrix[i][i] = 0.0);
    let rhs: Vector = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
    let solution = matrix.solve_ldl(&rhs).unwrap();
    Assert::default().eq_within_tols(&(matrix * &solution), &rhs)
}

#[test]
fn solve_ldl_vanishing_diagonal() -> Result<(), AssertionError> {
    let n = 6;
    let mut matrix = SquareMatrix::zero(n);
    for i in 0..n {
        for j in 0..i {
            let entry = ((i + j) % 5) as f64 + 1.0;
            matrix[i][j] = entry;
            matrix[j][i] = entry
        }
    }
    let rhs: Vector = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
    let solution = matrix.solve_ldl(&rhs).unwrap();
    Assert::default().eq_within_tols(&(matrix * &solution), &rhs)
}

#[test]
fn solve_ldl_scaled_dim_25() -> Result<(), AssertionError> {
    let scale = 1e-14;
    let rhs = rhs_dim_25();
    let solution = kkt_symmetric_dim_25().solve_ldl(&rhs).unwrap();
    let scaled = (kkt_symmetric_dim_25() * scale)
        .solve_ldl(&(&rhs * scale))
        .unwrap();
    Assert::default().eq_within_tols(&scaled, &solution)
}

fn diagonal(entries: &[f64]) -> SquareMatrix {
    let mut matrix = SquareMatrix::zero(entries.len());
    entries
        .iter()
        .enumerate()
        .for_each(|(i, &entry)| matrix[i][i] = entry);
    matrix
}

#[test]
fn inertia_diagonal() {
    let inertia = |entries: &[f64]| diagonal(entries).factorize_ldl().unwrap().inertia();
    assert_eq!(inertia(&[3.0, 2.0, 5.0]), (3, 0, 0));
    assert_eq!(inertia(&[-3.0, -2.0, -5.0]), (0, 3, 0));
    assert_eq!(inertia(&[3.0, -2.0, 5.0, -1.0]), (2, 2, 0))
}

#[test]
fn inertia_two_by_two_pivots() {
    let mut matrix = SquareMatrix::zero(4);
    matrix[0][1] = 1.0;
    matrix[1][0] = 1.0;
    matrix[2][3] = 2.0;
    matrix[3][2] = 2.0;
    assert_eq!(matrix.factorize_ldl().unwrap().inertia(), (2, 2, 0))
}

#[test]
fn inertia_kkt_minimum() {
    let matrix = SquareMatrix::from([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
    ]);
    assert_eq!(matrix.factorize_ldl().unwrap().inertia(), (3, 1, 0))
}

#[test]
fn inertia_kkt_saddle() {
    let matrix = SquareMatrix::from([[1.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]);
    assert_eq!(matrix.factorize_ldl().unwrap().inertia(), (1, 2, 0))
}

#[test]
fn inertia_kkt_dim_25() {
    let (positive, negative, zero) = kkt_symmetric_dim_25().factorize_ldl().unwrap().inertia();
    assert_eq!((positive + negative, zero), (25, 0))
}

#[test]
fn inertia_refactorized_in_place() {
    let mut decomposition = diagonal(&[1.0, 2.0, 3.0]).factorize_ldl().unwrap();
    assert_eq!(decomposition.inertia(), (3, 0, 0));
    let mut matrix = SquareMatrix::zero(3);
    matrix[0][1] = 1.0;
    matrix[1][0] = 1.0;
    matrix[2][2] = 4.0;
    matrix.factorize_ldl_into(&mut decomposition).unwrap();
    assert_eq!(decomposition.inertia(), (2, 1, 0));
    diagonal(&[-1.0, 2.0, -3.0])
        .factorize_ldl_into(&mut decomposition)
        .unwrap();
    assert_eq!(decomposition.inertia(), (1, 2, 0))
}
