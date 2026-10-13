use super::{
    MORE_SORENSEN_TOLERANCE, SparseError, SquareMatrix, Vector, more_sorensen_dense,
    more_sorensen_sparse,
};
use crate::math::sparse::SparseSolver;

fn bordered_pd() -> SquareMatrix {
    let mut matrix = SquareMatrix::zero(3);
    matrix[0][0] = 2.0;
    matrix[1][1] = 3.0;
    matrix[0][2] = 1.0;
    matrix[2][0] = 1.0;
    matrix[1][2] = 1.0;
    matrix[2][1] = 1.0;
    matrix
}

fn bordered_indefinite_primal() -> SquareMatrix {
    let mut matrix = SquareMatrix::zero(3);
    matrix[0][0] = -1.0;
    matrix[1][1] = -1.0;
    matrix[0][2] = 1.0;
    matrix[2][0] = 1.0;
    matrix[1][2] = 1.0;
    matrix[2][1] = 1.0;
    matrix
}

#[test]
fn interior_solution_matches_plain_solve() {
    let tangent = bordered_pd();
    let residual: Vector = [1.0, 1.0, 0.0].into_iter().collect();
    let step = more_sorensen_dense(&tangent, &residual, 1e3, 2);
    let plain = tangent.solve_ldl(&residual).unwrap();
    (0..3).for_each(|i| assert!((step[i] - plain[i]).abs() < 1e-8));
}

#[test]
fn boundary_solution_lands_on_the_radius() {
    let tangent = bordered_pd();
    let residual: Vector = [1.0, 0.0, 0.0].into_iter().collect();
    let radius = 0.05;
    let step = more_sorensen_dense(&tangent, &residual, radius, 2);
    let norm = (step[0] * step[0] + step[1] * step[1]).sqrt();
    assert!((norm - radius).abs() < 2.0 * MORE_SORENSEN_TOLERANCE * radius)
}

#[test]
fn handles_a_primal_block_that_starts_indefinite() {
    let tangent = bordered_indefinite_primal();
    let residual: Vector = [1.0, 1.0, 0.0].into_iter().collect();
    assert!(
        !tangent
            .factorize_ldl()
            .is_ok_and(|decomposition| decomposition.inertia() == (2, 1, 0)),
        "test premise: lambda = 0 should not already be admissible here"
    );
    let radius = 0.1;
    let step = more_sorensen_dense(&tangent, &residual, radius, 2);
    let norm = (step[0] * step[0] + step[1] * step[1]).sqrt();
    assert!(norm <= radius * (1.0 + 1e-6))
}

#[test]
fn unconstrained_degenerates_to_a_plain_shift() {
    let mut tangent = SquareMatrix::zero(2);
    tangent[0][0] = -1.0;
    tangent[1][1] = -2.0;
    let residual: Vector = [1.0, 1.0].into_iter().collect();
    let radius = 0.2;
    let step = more_sorensen_dense(&tangent, &residual, radius, 2);
    let norm = (step[0] * step[0] + step[1] * step[1]).sqrt();
    assert!((norm - radius).abs() < 2.0 * MORE_SORENSEN_TOLERANCE * radius)
}

#[test]
fn a_radius_just_inside_the_newton_step_is_not_overshot() {
    let tangent = bordered_pd();
    let residual: Vector = [1.0, 0.0, 0.0].into_iter().collect();
    let plain = tangent.solve_ldl(&residual).unwrap();
    let unshifted = (plain[0] * plain[0] + plain[1] * plain[1]).sqrt();
    let radius = 0.9 * unshifted;
    let step = more_sorensen_dense(&tangent, &residual, radius, 2);
    let norm = (step[0] * step[0] + step[1] * step[1]).sqrt();
    assert!(
        (norm - radius).abs() < 2.0 * MORE_SORENSEN_TOLERANCE * radius,
        "step of norm {norm} for a radius of {radius}"
    )
}

#[test]
fn an_indefinite_tangent_is_not_damped_to_its_scale() {
    let mut tangent = SquareMatrix::zero(2);
    tangent[0][0] = -1.0;
    tangent[1][1] = 1000.0;
    let residual: Vector = [1.0, 1.0].into_iter().collect();
    let radius = 0.5;
    let step = more_sorensen_dense(&tangent, &residual, radius, 2);
    let norm = (step[0] * step[0] + step[1] * step[1]).sqrt();
    assert!(
        (norm - radius).abs() < 2.0 * MORE_SORENSEN_TOLERANCE * radius,
        "step of norm {norm} for a radius of {radius}"
    )
}

fn bordered_pattern() -> Vec<(usize, usize)> {
    (0..3)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .filter(|&(i, j)| i == j || i == 2 || j == 2)
        .collect()
}

#[test]
fn sparse_matches_dense() {
    let tangent = bordered_pd();
    let residual: Vector = [1.0, 0.0, 0.0].into_iter().collect();
    let radius = 0.05;
    let solver = SparseSolver::from_pattern(3, bordered_pattern(), true);
    let sparse = more_sorensen_sparse(&solver, |i, j| tangent[i][j], &residual, radius, 2).unwrap();
    let dense = more_sorensen_dense(&tangent, &residual, radius, 2);
    (0..3).for_each(|i| assert!((sparse[i] - dense[i]).abs() < 1e-8))
}

#[test]
fn sparse_refuses_an_unsymmetric_solver() {
    let tangent = bordered_pd();
    let residual: Vector = [1.0, 0.0, 0.0].into_iter().collect();
    let solver = SparseSolver::from_pattern(3, bordered_pattern(), false);
    assert!(matches!(
        more_sorensen_sparse(&solver, |i, j| tangent[i][j], &residual, 0.05, 2),
        Err(SparseError::Unsymmetric)
    ))
}
