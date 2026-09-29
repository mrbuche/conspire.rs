use super::super::{SquareMatrix, Vector};
use crate::math::{
    Rank2,
    assert::{Assert, AssertionError},
};

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

fn floppy(scale: f64) -> SquareMatrix {
    let n = 6;
    let direction: Vec<f64> = (1..=n).map(|i| (i as f64).sqrt()).collect();
    let norm: f64 = direction.iter().map(|entry| entry * entry).sum();
    let reflection = |i: usize, j: usize| {
        (if i == j { 1.0 } else { 0.0 }) - 2.0 * direction[i] * direction[j] / norm
    };
    let mut matrix = SquareMatrix::zero(n);
    (0..n).for_each(|i| {
        (0..n).for_each(|j| {
            matrix[i][j] = scale
                * (0..n - 1)
                    .map(|k| (k + 1) as f64 * reflection(i, k) * reflection(j, k))
                    .sum::<f64>()
        })
    });
    matrix
}

#[test]
fn a_matrix_with_a_null_space_has_a_near_zero_pivot_at_any_scale() {
    [1e-3, 1.0, 1e9, 1e15].into_iter().for_each(|scale| {
        if let Ok(factorization) = floppy(scale).factorize_lu() {
            assert_eq!(factorization.near_zero_pivots(1e-10), 1, "scale {scale:e}")
        }
    })
}

#[test]
fn a_stiff_matrix_with_a_null_space_gets_through_the_factorization_but_not_the_count() {
    let factorization = floppy(1e12)
        .factorize_lu()
        .expect("rounding error above the absolute tolerance is not taken for singular");
    assert_eq!(factorization.near_zero_pivots(1e-10), 1);
}

#[test]
fn a_well_posed_matrix_has_no_near_zero_pivot_at_any_scale() {
    [1e-3, 1.0, 1e9, 1e15].into_iter().for_each(|scale| {
        let mut matrix = floppy(scale);
        (0..6).for_each(|i| matrix[i][i] += scale);
        let factorization = matrix.factorize_lu().unwrap();
        assert_eq!(factorization.near_zero_pivots(1e-10), 0, "scale {scale:e}")
    })
}

#[test]
fn an_ill_conditioned_matrix_is_not_mistaken_for_a_singular_one() {
    let mut matrix = SquareMatrix::zero(4);
    [1.0, 1e-2, 1e-4, 1e-6]
        .iter()
        .enumerate()
        .for_each(|(i, &entry)| matrix[i][i] = entry);
    assert_eq!(matrix.factorize_lu().unwrap().near_zero_pivots(1e-10), 0);
}

#[test]
fn solve_transpose_matches_the_solve_of_the_transposed_matrix() -> Result<(), AssertionError> {
    let matrix = kkt_dim_25();
    let mut skewed = matrix.clone();
    (0..25).for_each(|i| (0..25).for_each(|j| skewed[i][j] += 0.1 * ((i + 2 * j) % 5) as f64));
    let rhs: Vector = (0..25).map(|i| (i as f64).sin() + 0.5).collect();
    let factor = skewed.factorize_lu().unwrap();
    let expected = skewed.transpose().solve_lu(&rhs).unwrap();
    Assert::default().eq_within_tols(factor.solve_transpose(&rhs), &expected)
}
