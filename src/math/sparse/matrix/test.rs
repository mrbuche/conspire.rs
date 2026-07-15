use super::{CscMatrix, Scalar, Vector};
use crate::math::{
    Tensor,
    assert::{AssertionError, assert_eq},
};

const D: usize = 9;

fn dense() -> [[Scalar; D]; D] {
    [
        [2.0, 2.0, 4.0, 0.0, 0.0, 1.0, 0.0, 3.0, 3.0],
        [0.0, 3.0, 1.0, 0.0, 0.0, 1.0, 4.0, 0.0, 1.0],
        [3.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 4.0, 0.0],
        [4.0, 4.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 4.0],
        [0.0, 1.0, 0.0, 1.0, 1.0, 3.0, 0.0, 1.0, 1.0],
        [4.0, 2.0, 3.0, 0.0, 2.0, 4.0, 3.0, 0.0, 4.0],
        [1.0, 3.0, 2.0, 0.0, 0.0, 0.0, 2.0, 4.0, 2.0],
        [0.0, 2.0, 2.0, 4.0, 1.0, 0.0, 4.0, 2.0, 2.0],
        [1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 4.0, 0.0, 1.0],
    ]
}

fn vector() -> Vector {
    Vector::from([2.0, 1.0, 3.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0])
}

fn pattern() -> Vec<(usize, usize)> {
    let matrix = dense();
    let mut pattern: Vec<(usize, usize)> = (0..D)
        .flat_map(|i| (0..D).filter(move |&j| i != j).map(move |j| (i, j)))
        .filter(|&(i, j)| matrix[i][j] != 0.0)
        .collect();
    pattern.reverse();
    (0..D).for_each(|i| pattern.push((i, i)));
    pattern
}

fn sparse() -> CscMatrix {
    let matrix = dense();
    let mut sparse = CscMatrix::from_pattern(D, D, pattern());
    sparse.fill(|i, j| matrix[i][j]);
    sparse
}

fn assert_eq_dense(sparse: &CscMatrix, dense: &[[Scalar; D]; D]) {
    assert_eq!(sparse.height(), D);
    assert_eq!(sparse.width(), D);
    let mut entries = sparse.iter();
    (0..D).for_each(|j| {
        (0..D).for_each(|i| {
            if dense[i][j] != 0.0 {
                assert_eq!(entries.next(), Some((i, j, &dense[i][j])));
            }
        })
    });
    assert!(entries.next().is_none());
}

#[test]
fn from_pattern_and_fill() {
    assert_eq_dense(&sparse(), &dense())
}

#[test]
fn fill_sums_duplicates() {
    let matrix = dense();
    let mut pattern = pattern();
    pattern.push((3, 0));
    pattern.push((3, 0));
    let mut sparse = CscMatrix::from_pattern(D, D, pattern);
    sparse.fill(|i, j| matrix[i][j] / if (i, j) == (3, 0) { 3.0 } else { 1.0 });
    assert_eq_dense(&sparse, &matrix)
}

#[test]
fn multiply_vector() -> Result<(), AssertionError> {
    let vector = vector();
    let product: Vector = dense()
        .iter()
        .map(|dense_i| {
            dense_i
                .iter()
                .zip(vector.iter())
                .map(|(dense_ij, vector_j)| dense_ij * vector_j)
                .sum()
        })
        .collect();
    assert_eq(&(&sparse() * &vector), &product)
}

#[test]
fn nonzeros() {
    assert_eq!(sparse().nonzeros(), pattern().len());
}

#[test]
fn refill() {
    let matrix = dense();
    let mut doubled = matrix;
    doubled
        .iter_mut()
        .for_each(|row| row.iter_mut().for_each(|entry| *entry *= 2.0));
    let mut sparse = sparse();
    sparse.fill(|i, j| 2.0 * matrix[i][j]);
    assert_eq_dense(&sparse, &doubled)
}

#[test]
fn transpose() {
    let matrix = dense();
    let mut transposed = matrix;
    (0..D).for_each(|i| (0..D).for_each(|j| transposed[i][j] = matrix[j][i]));
    let sparse = sparse();
    let transpose = sparse.transpose();
    assert_eq_dense(&transpose, &transposed);
    assert_eq!(transpose.transpose(), sparse);
}
