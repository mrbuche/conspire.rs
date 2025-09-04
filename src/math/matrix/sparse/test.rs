use crate::math::{SparseMatrix, SquareMatrix, Tensor, TestError, assert_eq, assert_eq_within_tols};
use super::super::square::test::{get_solve_lu, square_matrix_dim_9, vector_dim_9};

#[test]
fn mul() -> Result<(), TestError> {
    let mut matrix_a = SparseMatrix::new(3, 3);
    matrix_a.add_value(0, 0, 1.0);
    matrix_a.add_value(0, 2, 2.0);
    matrix_a.add_value(1, 1, 3.0);
    matrix_a.add_value(2, 0, 4.0);
    matrix_a.add_value(2, 2, 5.0);
    let mut matrix_b = SparseMatrix::new(3, 3);
    matrix_b.add_value(0, 0, 6.0);
    matrix_b.add_value(0, 1, 7.0);
    matrix_b.add_value(1, 1, 8.0);
    matrix_b.add_value(2, 0, 9.0);
    matrix_b.add_value(2, 2, 10.0);
    let matrix_c = matrix_a * &matrix_b;
    SquareMatrix::new(&[&[24.0, 7.0, 20.0], &[0.0, 24.0, 0.0], &[69.0, 28.0, 50.0]])
        .iter()
        .enumerate()
        .try_for_each(|(i, c_i)| {
            c_i.iter()
                .enumerate()
                .try_for_each(|(j, c_ij)| assert_eq(c_ij, &matrix_c[[i, j]]))
        })
}

#[test]
fn solve_lu() -> Result<(), TestError> {
    println!("{:?}", square_matrix_dim_9().solve_lu(&vector_dim_9()));
    assert_eq_within_tols(
        // &square_matrix_dim_9().solve_lu(&vector_dim_9())?,
        &square_matrix_dim_9().solve_lu(&vector_dim_9()),
        &get_solve_lu(),
    )
}

use crate::math::TensorVec;

#[test]
fn foo() {
    let time = std::time::Instant::now();
    let mut matrix_a = SparseMatrix::new(1000, 1000);
    matrix_a.add_value(0, 0, 1.0);
    matrix_a.add_value(0, 2, 2.0);
    matrix_a.add_value(1, 1, 3.0);
    matrix_a.add_value(2, 0, 4.0);
    matrix_a.add_value(2, 2, 5.0);
    let mut matrix_b = SparseMatrix::new(1000, 1000);
    matrix_b.add_value(0, 0, 6.0);
    matrix_b.add_value(0, 1, 7.0);
    matrix_b.add_value(1, 1, 8.0);
    matrix_b.add_value(2, 0, 9.0);
    matrix_b.add_value(2, 2, 10.0);
    let matrix_c = matrix_a * &matrix_b;
    println!("Matrix C_00: {}", matrix_c[[0, 0]]);
    println!("{:?}", time.elapsed());
    let time = std::time::Instant::now();
    let matrix_a = SquareMatrix::zero(1000);
    let matrix_b = SquareMatrix::zero(1000);
    let matrix_c = matrix_a * matrix_b;
    println!("Matrix C_00: {}", matrix_c[0][0]);
    println!("{:?}", time.elapsed());
}
