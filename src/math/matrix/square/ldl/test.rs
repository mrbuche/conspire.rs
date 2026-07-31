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
    // every third diagonal vanishes, so no one-by-one pivot is available there
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
    // the diagonal vanishes entirely, so every pivot must be two-by-two
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
