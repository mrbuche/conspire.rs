use super::{super::Matrix, SquareMatrix, Vector};
use crate::math::assert::Assert;
use crate::math::assert::AssertionError;

fn vector_dim_6() -> Vector {
    Vector::from([2.0, 1.0, 3.0, 2.0, 1.0, 3.0])
}

fn matrix_dim_6_9() -> Matrix {
    [
        Vector::from([2.0, 2.0, 4.0, 0.0, 0.0, 1.0, 1.0, 3.0, 3.0]),
        Vector::from([0.0, 3.0, 1.0, 0.0, 0.0, 1.0, 4.0, 2.0, 1.0]),
        Vector::from([3.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 4.0, 2.0]),
        Vector::from([4.0, 4.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 4.0]),
        Vector::from([0.0, 1.0, 0.0, 1.0, 1.0, 3.0, 0.0, 1.0, 1.0]),
        Vector::from([4.0, 2.0, 3.0, 4.0, 2.0, 4.0, 3.0, 0.0, 4.0]),
    ]
    .into_iter()
    .collect()
}

fn square_matrix_dim_9() -> SquareMatrix {
    SquareMatrix::from([
        [2.0, 2.0, 4.0, 0.0, 0.0, 1.0, 1.0, 3.0, 3.0],
        [0.0, 3.0, 1.0, 0.0, 0.0, 1.0, 4.0, 2.0, 1.0],
        [3.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 4.0, 2.0],
        [4.0, 4.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 4.0],
        [0.0, 1.0, 0.0, 1.0, 1.0, 3.0, 0.0, 1.0, 1.0],
        [4.0, 2.0, 3.0, 4.0, 2.0, 4.0, 3.0, 0.0, 4.0],
        [1.0, 3.0, 2.0, 0.0, 0.0, 0.0, 2.0, 4.0, 2.0],
        [2.0, 2.0, 2.0, 4.0, 1.0, 2.0, 4.0, 2.0, 2.0],
        [1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 4.0, 2.0, 1.0],
    ])
}

fn other_square_matrix_dim_9() -> SquareMatrix {
    SquareMatrix::from([
        [0.0, 4.0, 2.0, 0.0, 1.0, 4.0, 2.0, 4.0, 1.0],
        [1.0, 2.0, 2.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0],
        [3.0, 0.0, 2.0, 3.0, 3.0, 0.0, 0.0, 0.0, 2.0],
        [2.0, 3.0, 0.0, 0.0, 1.0, 3.0, 3.0, 4.0, 2.0],
        [0.0, 4.0, 1.0, 3.0, 1.0, 1.0, 1.0, 2.0, 1.0],
        [1.0, 3.0, 0.0, 3.0, 3.0, 2.0, 1.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 1.0, 3.0, 4.0],
        [2.0, 0.0, 4.0, 3.0, 1.0, 2.0, 0.0, 3.0, 4.0],
        [4.0, 2.0, 0.0, 0.0, 4.0, 0.0, 4.0, 2.0, 2.0],
    ])
}

fn get_vector_mul_matrix_dim_6_9() -> Vector {
    Vector::from([33.0, 22.0, 21.0, 23.0, 9.0, 29.0, 27.0, 21.0, 34.0])
}

fn get_square_matrix_mul_other_square_matrix_dim_9() -> SquareMatrix {
    SquareMatrix::from([
        [33.0, 21.0, 28.0, 27.0, 32.0, 25.0, 18.0, 33.0, 36.0],
        [15.0, 11.0, 16.0, 19.0, 12.0, 27.0, 9.0, 29.0, 32.0],
        [26.0, 31.0, 24.0, 28.0, 29.0, 44.0, 27.0, 57.0, 57.0],
        [25.0, 45.0, 17.0, 10.0, 26.0, 37.0, 32.0, 45.0, 21.0],
        [12.0, 20.0, 7.0, 16.0, 16.0, 15.0, 11.0, 22.0, 21.0],
        [39.0, 60.0, 20.0, 32.0, 47.0, 53.0, 45.0, 69.0, 56.0],
        [25.0, 14.0, 28.0, 23.0, 19.0, 27.0, 12.0, 32.0, 33.0],
        [30.0, 38.0, 21.0, 27.0, 29.0, 47.0, 31.0, 58.0, 51.0],
        [28.0, 25.0, 20.0, 24.0, 23.0, 40.0, 23.0, 47.0, 45.0],
    ])
}

#[test]
fn vector_mul_matrix_dim_6_9() -> Result<(), AssertionError> {
    Assert::eq(
        &(&vector_dim_6() * &matrix_dim_6_9()),
        &get_vector_mul_matrix_dim_6_9(),
    )
}

#[test]
fn square_matrix_mul_other_square_matrix_dim_9() -> Result<(), AssertionError> {
    Assert::eq(
        &(square_matrix_dim_9() * other_square_matrix_dim_9()),
        &get_square_matrix_mul_other_square_matrix_dim_9(),
    )
}

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
