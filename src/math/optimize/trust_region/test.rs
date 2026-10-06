use super::{MORE_SORENSEN_TOLERANCE, SquareMatrix, Vector, more_sorensen_dense};

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
