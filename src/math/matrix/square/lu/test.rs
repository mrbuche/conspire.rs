use super::super::{SquareMatrix, Vector};
use crate::math::assert::{Assert, AssertionError};

fn kkt_dim_25() -> SquareMatrix {
    let (ng, cg, nl, cl) = (9, 4, 9, 3);
    let (no, n) = (ng + cg, ng + cg + nl + cl);
    let mut matrix = SquareMatrix::zero(n);
    for i in 0..ng {
        for j in 0..ng {
            let entry = ((i * 13 + j * 7) % 11) as f64 - 5.0;
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

#[test]
fn solve_lu_kkt_dim_25() -> Result<(), AssertionError> {
    let matrix = kkt_dim_25();
    let rhs: Vector = (0..25).map(|i| ((i * 5) % 7) as f64 - 3.0).collect();
    let solution = matrix.solve_lu(&rhs).unwrap();
    Assert::default().eq_within_tols(&(matrix * &solution), &rhs)
}

#[test]
fn solve_lu_scaled_dim_25() -> Result<(), AssertionError> {
    let rhs: Vector = (0..25).map(|i| ((i * 5) % 7) as f64 - 3.0).collect();
    let solution = kkt_dim_25().solve_lu(&rhs).unwrap();
    let scale = 1e-14;
    let scaled = (kkt_dim_25() * scale).solve_lu(&(&rhs * scale)).unwrap();
    Assert::default().eq_within_tols(&scaled, &solution)
}
