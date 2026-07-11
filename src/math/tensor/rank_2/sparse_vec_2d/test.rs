use super::{TensorRank2, TensorRank2SparseVec2D};
use crate::math::{Hessian, Rank2, SquareMatrix};

fn block(value: f64) -> TensorRank2<2, 1, 1> {
    TensorRank2::from([[value, 2.0 * value], [3.0 * value, 4.0 * value]])
}

fn accumulator() -> TensorRank2SparseVec2D<2, 1, 1> {
    let mut stiffnesses = TensorRank2SparseVec2D::<2, 1, 1>::zero(3);
    stiffnesses[2][0] += block(1.0);
    stiffnesses[0][0] += block(2.0);
    stiffnesses[0][2] += block(3.0);
    stiffnesses[0][0] += block(4.0);
    stiffnesses[1][1] += block(5.0);
    stiffnesses
}

fn dense() -> SquareMatrix {
    let mut square_matrix = SquareMatrix::zero(6);
    accumulator().fill_into(&mut square_matrix);
    square_matrix
}

#[test]
fn insertion_sums_and_sorts() {
    let stiffnesses = accumulator();
    assert_eq!(stiffnesses[0][0][0][0], 6.0);
    assert_eq!(stiffnesses[0][2][1][0], 9.0);
    assert_eq!(stiffnesses[2][0][1][1], 4.0);
    let columns: Vec<usize> = stiffnesses[0]
        .entries()
        .map(|&(column, _)| column)
        .collect();
    assert_eq!(columns, [0, 2]);
}

#[test]
fn fill_into_scatters_blocks() {
    let square_matrix = dense();
    assert_eq!(square_matrix[0][0], 6.0);
    assert_eq!(square_matrix[1][0], 18.0);
    assert_eq!(square_matrix[0][5], 6.0);
    assert_eq!(square_matrix[4][0], 1.0);
    assert_eq!(square_matrix[2][2], 5.0);
    assert_eq!(square_matrix[0][2], 0.0);
}

#[test]
fn retain_from_filters_and_remaps() {
    let retained = [true, false, true, true, true, false];
    let square_matrix = accumulator().retain_from(&retained);
    let full = dense();
    let kept: Vec<usize> = (0..6).filter(|&p| retained[p]).collect();
    kept.iter().enumerate().for_each(|(p, &full_p)| {
        kept.iter()
            .enumerate()
            .for_each(|(q, &full_q)| assert_eq!(square_matrix[p][q], full[full_p][full_q]))
    });
}

#[test]
fn rotation_preserves_structure() {
    let rotation = TensorRank2::<2, 1, 1>::from([[0.0, -1.0], [1.0, 0.0]]);
    let rotated = rotation.transpose() * accumulator() * rotation;
    let mut square_matrix = SquareMatrix::zero(6);
    rotated.fill_into(&mut square_matrix);
    let full = dense();
    assert_eq!(square_matrix[0][0], full[1][1]);
    assert_eq!(square_matrix[1][0], -full[0][1]);
    assert_eq!(square_matrix[0][5], -full[1][4]);
}

#[test]
fn merge_add_and_subtract() {
    let mut other = TensorRank2SparseVec2D::<2, 1, 1>::zero(3);
    other[0][1] += block(7.0);
    other[0][0] += block(1.0);
    let sum: Vec<_> = accumulator()
        .into_iter()
        .zip(other.clone())
        .map(|(row, other_row)| row + other_row)
        .collect();
    assert_eq!(sum[0][0][0][0], 7.0);
    assert_eq!(sum[0][1][0][0], 7.0);
    assert_eq!(sum[0][2][0][0], 3.0);
    let difference: Vec<_> = accumulator()
        .into_iter()
        .zip(other)
        .map(|(row, other_row)| row - other_row)
        .collect();
    assert_eq!(difference[0][0][0][0], 5.0);
    assert_eq!(difference[0][1][0][0], -7.0);
}
