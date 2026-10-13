use crate::math::{
    Hessian, Matrix, SquareMatrix, Vector,
    assert::{Assert, AssertionError},
};

fn vector_dim_6() -> Vector {
    [2.0, 1.0, 3.0, 2.0, 1.0, 3.0].into()
}

fn matrix_dim_6_9() -> Matrix {
    [
        [2.0, 2.0, 4.0, 0.0, 0.0, 1.0, 1.0, 3.0, 3.0].into(),
        [0.0, 3.0, 1.0, 0.0, 0.0, 1.0, 4.0, 2.0, 1.0].into(),
        [3.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 4.0, 2.0].into(),
        [4.0, 4.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 4.0].into(),
        [0.0, 1.0, 0.0, 1.0, 1.0, 3.0, 0.0, 1.0, 1.0].into(),
        [4.0, 2.0, 3.0, 4.0, 2.0, 4.0, 3.0, 0.0, 4.0].into(),
    ]
    .into_iter()
    .collect()
}

fn square_matrix_dim_9() -> SquareMatrix {
    [
        [2.0, 2.0, 4.0, 0.0, 0.0, 1.0, 1.0, 3.0, 3.0],
        [0.0, 3.0, 1.0, 0.0, 0.0, 1.0, 4.0, 2.0, 1.0],
        [3.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 4.0, 2.0],
        [4.0, 4.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 4.0],
        [0.0, 1.0, 0.0, 1.0, 1.0, 3.0, 0.0, 1.0, 1.0],
        [4.0, 2.0, 3.0, 4.0, 2.0, 4.0, 3.0, 0.0, 4.0],
        [1.0, 3.0, 2.0, 0.0, 0.0, 0.0, 2.0, 4.0, 2.0],
        [2.0, 2.0, 2.0, 4.0, 1.0, 2.0, 4.0, 2.0, 2.0],
        [1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 4.0, 2.0, 1.0],
    ]
    .into()
}

fn other_square_matrix_dim_9() -> SquareMatrix {
    [
        [0.0, 4.0, 2.0, 0.0, 1.0, 4.0, 2.0, 4.0, 1.0],
        [1.0, 2.0, 2.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0],
        [3.0, 0.0, 2.0, 3.0, 3.0, 0.0, 0.0, 0.0, 2.0],
        [2.0, 3.0, 0.0, 0.0, 1.0, 3.0, 3.0, 4.0, 2.0],
        [0.0, 4.0, 1.0, 3.0, 1.0, 1.0, 1.0, 2.0, 1.0],
        [1.0, 3.0, 0.0, 3.0, 3.0, 2.0, 1.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 1.0, 3.0, 4.0],
        [2.0, 0.0, 4.0, 3.0, 1.0, 2.0, 0.0, 3.0, 4.0],
        [4.0, 2.0, 0.0, 0.0, 4.0, 0.0, 4.0, 2.0, 2.0],
    ]
    .into()
}

fn get_vector_mul_matrix_dim_6_9() -> Vector {
    [33.0, 22.0, 21.0, 23.0, 9.0, 29.0, 27.0, 21.0, 34.0].into()
}

fn get_square_matrix_mul_other_square_matrix_dim_9() -> SquareMatrix {
    [
        [33.0, 21.0, 28.0, 27.0, 32.0, 25.0, 18.0, 33.0, 36.0],
        [15.0, 11.0, 16.0, 19.0, 12.0, 27.0, 9.0, 29.0, 32.0],
        [26.0, 31.0, 24.0, 28.0, 29.0, 44.0, 27.0, 57.0, 57.0],
        [25.0, 45.0, 17.0, 10.0, 26.0, 37.0, 32.0, 45.0, 21.0],
        [12.0, 20.0, 7.0, 16.0, 16.0, 15.0, 11.0, 22.0, 21.0],
        [39.0, 60.0, 20.0, 32.0, 47.0, 53.0, 45.0, 69.0, 56.0],
        [25.0, 14.0, 28.0, 23.0, 19.0, 27.0, 12.0, 32.0, 33.0],
        [30.0, 38.0, 21.0, 27.0, 29.0, 47.0, 31.0, 58.0, 51.0],
        [28.0, 25.0, 20.0, 24.0, 23.0, 40.0, 23.0, 47.0, 45.0],
    ]
    .into()
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

#[test]
fn quadratic_form_matches_multiplication() {
    let mut square_matrix = SquareMatrix::zero(3);
    (0..3).for_each(|i| (0..3).for_each(|j| square_matrix[i][j] = (1 + i * 3 + j) as f64));
    let vector = Vector::from([1.0, -2.0, 3.0]);
    let form = square_matrix.quadratic_form(&vector);
    Assert::default()
        .eq_within_tols(form, &(&vector * (square_matrix * &vector)))
        .unwrap()
}

#[test]
fn quadratic_form_leading_ignores_trailing_rows_and_columns() {
    let mut square_matrix = SquareMatrix::zero(3);
    (0..3).for_each(|i| (0..3).for_each(|j| square_matrix[i][j] = (1 + i * 3 + j) as f64));
    let vector = Vector::from([1.0, -2.0, 3.0]);
    Assert::default()
        .eq_within_tols(square_matrix.quadratic_form_leading(&vector, 2), &9.0)
        .unwrap();
    Assert::default()
        .eq_within_tols(
            square_matrix.quadratic_form_leading(&vector, 3),
            &square_matrix.quadratic_form(&vector),
        )
        .unwrap()
}

#[test]
fn multiplication_by_reference_matches_multiplication_by_value() {
    let mut square_matrix = SquareMatrix::zero(3);
    (0..3).for_each(|i| (0..3).for_each(|j| square_matrix[i][j] = (1 + i * 3 + j) as f64));
    let vector = Vector::from([1.0, -2.0, 3.0]);
    let expected = square_matrix.clone() * &vector;
    assert_eq!(&square_matrix * &vector, expected);
    assert_eq!(&square_matrix * vector.clone(), expected);
    assert_eq!(expected, Vector::from([6.0, 12.0, 18.0]));
}
