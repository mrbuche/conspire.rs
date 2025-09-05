use super::{super::Matrix, Banded, SquareMatrix, TensorVec, Vector};
use crate::math::test::{TestError, assert_eq, assert_eq_within_tols};

fn vector_dim_6() -> Vector {
    Vector::new(&[2.0, 1.0, 3.0, 2.0, 1.0, 3.0])
}

pub fn vector_dim_9() -> Vector {
    Vector::new(&[2.0, 1.0, 3.0, 2.0, 1.0, 3.0, 5.0, 1.0, 2.0])
}

fn matrix_dim_6_9() -> Matrix {
    [
        Vector::new(&[2.0, 2.0, 4.0, 0.0, 0.0, 1.0, 1.0, 3.0, 3.0]),
        Vector::new(&[0.0, 3.0, 1.0, 0.0, 0.0, 1.0, 4.0, 2.0, 1.0]),
        Vector::new(&[3.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 4.0, 2.0]),
        Vector::new(&[4.0, 4.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 4.0]),
        Vector::new(&[0.0, 1.0, 0.0, 1.0, 1.0, 3.0, 0.0, 1.0, 1.0]),
        Vector::new(&[4.0, 2.0, 3.0, 4.0, 2.0, 4.0, 3.0, 0.0, 4.0]),
    ]
    .into_iter()
    .collect()
}

pub fn square_matrix_dim_9() -> SquareMatrix {
    SquareMatrix::new(&[
        &[2.0, 2.0, 4.0, 0.0, 0.0, 1.0, 1.0, 3.0, 3.0],
        &[0.0, 3.0, 1.0, 0.0, 0.0, 1.0, 4.0, 2.0, 1.0],
        &[3.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 4.0, 2.0],
        &[4.0, 4.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 4.0],
        &[0.0, 1.0, 0.0, 1.0, 1.0, 3.0, 0.0, 1.0, 1.0],
        &[4.0, 2.0, 3.0, 4.0, 2.0, 4.0, 3.0, 0.0, 4.0],
        &[1.0, 3.0, 2.0, 0.0, 0.0, 0.0, 2.0, 4.0, 2.0],
        &[2.0, 2.0, 2.0, 4.0, 1.0, 2.0, 4.0, 2.0, 2.0],
        &[1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 4.0, 2.0, 1.0],
    ])
}

fn other_square_matrix_dim_9() -> SquareMatrix {
    SquareMatrix::new(&[
        &[0.0, 4.0, 2.0, 0.0, 1.0, 4.0, 2.0, 4.0, 1.0],
        &[1.0, 2.0, 2.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0],
        &[3.0, 0.0, 2.0, 3.0, 3.0, 0.0, 0.0, 0.0, 2.0],
        &[2.0, 3.0, 0.0, 0.0, 1.0, 3.0, 3.0, 4.0, 2.0],
        &[0.0, 4.0, 1.0, 3.0, 1.0, 1.0, 1.0, 2.0, 1.0],
        &[1.0, 3.0, 0.0, 3.0, 3.0, 2.0, 1.0, 3.0, 4.0],
        &[0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 1.0, 3.0, 4.0],
        &[2.0, 0.0, 4.0, 3.0, 1.0, 2.0, 0.0, 3.0, 4.0],
        &[4.0, 2.0, 0.0, 0.0, 4.0, 0.0, 4.0, 2.0, 2.0],
    ])
}

fn get_vector_mul_matrix_dim_6_9() -> Vector {
    Vector::new(&[33.0, 22.0, 21.0, 23.0, 9.0, 29.0, 27.0, 21.0, 34.0])
}

fn get_square_matrix_mul_other_square_matrix_dim_9() -> SquareMatrix {
    SquareMatrix::new(&[
        &[33.0, 21.0, 28.0, 27.0, 32.0, 25.0, 18.0, 33.0, 36.0],
        &[15.0, 11.0, 16.0, 19.0, 12.0, 27.0, 9.0, 29.0, 32.0],
        &[26.0, 31.0, 24.0, 28.0, 29.0, 44.0, 27.0, 57.0, 57.0],
        &[25.0, 45.0, 17.0, 10.0, 26.0, 37.0, 32.0, 45.0, 21.0],
        &[12.0, 20.0, 7.0, 16.0, 16.0, 15.0, 11.0, 22.0, 21.0],
        &[39.0, 60.0, 20.0, 32.0, 47.0, 53.0, 45.0, 69.0, 56.0],
        &[25.0, 14.0, 28.0, 23.0, 19.0, 27.0, 12.0, 32.0, 33.0],
        &[30.0, 38.0, 21.0, 27.0, 29.0, 47.0, 31.0, 58.0, 51.0],
        &[28.0, 25.0, 20.0, 24.0, 23.0, 40.0, 23.0, 47.0, 45.0],
    ])
}

pub fn get_solve_lu() -> Vector {
    Vector::new(&[
        16.870725604670554,
        8.541284403669726,
        5.326105087572977,
        -4.495412844036696,
        10.014178482068388,
        3.4787322768974156,
        -3.1184320266889083,
        2.541284403669724,
        -26.037531276063383,
    ])
}

#[test]
fn vector_mul_matrix_dim_6_9() -> Result<(), TestError> {
    assert_eq(
        &(&vector_dim_6() * &matrix_dim_6_9()),
        &get_vector_mul_matrix_dim_6_9(),
    )
}

#[test]
fn square_matrix_mul_other_square_matrix_dim_9() -> Result<(), TestError> {
    assert_eq(
        &(square_matrix_dim_9() * other_square_matrix_dim_9()),
        &get_square_matrix_mul_other_square_matrix_dim_9(),
    )
}

#[test]
fn solve_lu() -> Result<(), TestError> {
    assert_eq_within_tols(
        // &square_matrix_dim_9().solve_lu(&vector_dim_9())?,
        &square_matrix_dim_9().solve_lu(&vector_dim_9()).unwrap(),
        &get_solve_lu(),
    )
}

#[test]
fn solve_lu_banded() -> Result<(), TestError> {
    assert_eq_within_tols(
        // &square_matrix_dim_9().solve_lu(&vector_dim_9())?,
        &square_matrix_dim_9()
            .solve_lu_banded(
                &vector_dim_9(),
                &Banded {
                    bandwidth: 9,
                    inverse: (0..9).collect(),
                    mapping: (0..9).collect(),
                },
            )
            .unwrap(),
        &get_solve_lu(),
    )
}
