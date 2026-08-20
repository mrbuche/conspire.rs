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
fn inertia_positive_definite() {
    let mut matrix = SquareMatrix::zero(3);
    (0..3).for_each(|i| matrix[i][i] = (i + 1) as f64);
    assert_eq!(matrix.factorize_ldl().unwrap().inertia(), (3, 0, 0))
}

#[test]
fn inertia_negative_definite() {
    let mut matrix = SquareMatrix::zero(3);
    (0..3).for_each(|i| matrix[i][i] = -((i + 1) as f64));
    assert_eq!(matrix.factorize_ldl().unwrap().inertia(), (0, 3, 0))
}

#[test]
fn inertia_indefinite_two_by_two() {
    let mut matrix = SquareMatrix::zero(2);
    matrix[0][1] = 1.0;
    matrix[1][0] = 1.0;
    assert_eq!(matrix.factorize_ldl().unwrap().inertia(), (1, 1, 0))
}

#[test]
fn inertia_kkt_bordered_hessian() {
    let mut matrix = SquareMatrix::zero(3);
    matrix[0][0] = 2.0;
    matrix[1][1] = 3.0;
    matrix[0][2] = 1.0;
    matrix[2][0] = 1.0;
    matrix[1][2] = 1.0;
    matrix[2][1] = 1.0;
    assert_eq!(matrix.factorize_ldl().unwrap().inertia(), (2, 1, 0))
}

#[test]
fn inertia_kkt_dim_25() {
    assert_eq!(
        kkt_symmetric_dim_25().factorize_ldl().unwrap().inertia(),
        (12, 13, 0)
    )
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
