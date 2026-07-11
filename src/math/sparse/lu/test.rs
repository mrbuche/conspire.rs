use super::{CscMatrix, SparseError, Vector};
use crate::math::test::{TestError, assert_eq_within_tols};

fn matrix_dim_9() -> CscMatrix {
    let dense = [
        [2.0, 2.0, 4.0, 0.0, 0.0, 1.0, 0.0, 3.0, 3.0],
        [0.0, 3.0, 1.0, 0.0, 0.0, 1.0, 4.0, 0.0, 1.0],
        [3.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 4.0, 0.0],
        [4.0, 4.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 4.0],
        [0.0, 1.0, 0.0, 1.0, 1.0, 3.0, 0.0, 1.0, 1.0],
        [4.0, 2.0, 3.0, 0.0, 2.0, 4.0, 3.0, 0.0, 4.0],
        [1.0, 3.0, 2.0, 0.0, 0.0, 0.0, 2.0, 4.0, 2.0],
        [0.0, 2.0, 2.0, 4.0, 1.0, 0.0, 4.0, 2.0, 2.0],
        [1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 4.0, 0.0, 1.0],
    ];
    let pattern = (0..9)
        .flat_map(|i| (0..9).map(move |j| (i, j)))
        .filter(|&(i, j)| dense[i][j] != 0.0)
        .collect();
    let mut matrix = CscMatrix::from_pattern(9, 9, pattern);
    matrix.fill(|i, j| dense[i][j]);
    matrix
}

fn matrix_dim_100() -> CscMatrix {
    let n = 100;
    let mut pattern: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
    (0..n).for_each(|i| {
        pattern.push((i, (3 * i + 1) % n));
        pattern.push(((5 * i + 2) % n, i));
        pattern.push((i, (i + 7) % n));
    });
    let mut seed = 1_u64;
    let mut random = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f64 / (1u64 << 31) as f64 - 1.0
    };
    let mut matrix = CscMatrix::from_pattern(n, n, pattern);
    matrix.fill(|i, j| if i == j { 8.0 } else { random() });
    matrix
}

fn assert_solves(matrix: &CscMatrix, b: &Vector) -> Result<(), TestError> {
    let lu = matrix.lu().map_err(|_| TestError {
        message: "Factorization failed.".to_string(),
    })?;
    assert_eq_within_tols(&(matrix * &lu.solve(b)), b)
}

#[test]
fn solve_dim_9() -> Result<(), TestError> {
    assert_solves(
        &matrix_dim_9(),
        &Vector::from([2.0, 1.0, 3.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0]),
    )
}

#[test]
fn solve_dim_100() -> Result<(), TestError> {
    let matrix = matrix_dim_100();
    let b = (0..100).map(|i| (i % 13) as f64 - 6.0).collect();
    assert_solves(&matrix, &b)
}

#[test]
fn structurally_singular() {
    let pattern = vec![(0, 0), (1, 0), (1, 1), (2, 2), (0, 2)];
    let mut matrix = CscMatrix::from_pattern(4, 4, pattern);
    matrix.fill(|_, _| 1.0);
    assert!(
        matrix
            .lu()
            .is_err_and(|error| error == SparseError::Singular)
    )
}

#[test]
fn numerically_singular() {
    let mut matrix = CscMatrix::from_pattern(2, 2, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    matrix.fill(|_, _| 1.0);
    assert!(
        matrix
            .lu()
            .is_err_and(|error| error == SparseError::Singular)
    )
}

#[test]
fn fill_in() {
    let lu = matrix_dim_100().lu().expect("Factorization failed.");
    assert!(lu.nonzeros() >= matrix_dim_100().nonzeros());
}
