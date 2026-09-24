use super::{Krylov, KrylovMethod};
use crate::math::{SquareMatrix, Vector, assert::Assert, optimize::Preconditioning};

fn krylov() -> Krylov {
    Krylov {
        rel_tol: 1e-15,
        ..Default::default()
    }
}

fn minres() -> Krylov {
    Krylov {
        method: KrylovMethod::Minres,
        rel_tol: 1e-15,
        ..Default::default()
    }
}

/// Positive definite, and coupled off the diagonal.
fn coupled() -> SquareMatrix {
    let mut matrix = SquareMatrix::zero(3);
    matrix[0][0] = 4.0;
    matrix[1][1] = 3.0;
    matrix[2][2] = 5.0;
    matrix[0][1] = 1.0;
    matrix[1][0] = 1.0;
    matrix[1][2] = 2.0;
    matrix[2][1] = 2.0;
    matrix
}

/// Symmetric, and indefinite: the leading block is positive and the trailing
/// one negative, which is the shape a constraint gives a system.
fn indefinite() -> SquareMatrix {
    let mut matrix = SquareMatrix::zero(3);
    matrix[0][0] = 2.0;
    matrix[1][1] = 3.0;
    matrix[0][2] = 1.0;
    matrix[2][0] = 1.0;
    matrix[1][2] = 1.0;
    matrix[2][1] = 1.0;
    matrix
}

fn right_hand_side() -> Vector {
    [1.0, 2.0, 3.0].into_iter().collect()
}

fn apply(matrix: &SquareMatrix) -> impl FnMut(&Vector) -> Vector + '_ {
    |direction| matrix.clone() * direction
}

#[test]
fn matches_the_direct_solve() {
    let matrix = coupled();
    let rhs = right_hand_side();
    let expected = matrix.clone().solve_lu(&rhs).unwrap();
    let solution = krylov()
        .solve_operator(apply(&matrix), Preconditioning::None, &rhs)
        .unwrap();
    Assert::default()
        .eq_within_tols(&solution, &expected)
        .unwrap();
}

#[test]
fn diagonal_preconditioner_also_matches() {
    let matrix = coupled();
    let rhs = right_hand_side();
    let expected = matrix.clone().solve_lu(&rhs).unwrap();
    let diagonal: Vector = (0..3).map(|i| matrix[i][i]).collect();
    let solution = krylov()
        .solve_operator(apply(&matrix), Preconditioning::Diagonal(diagonal), &rhs)
        .unwrap();
    Assert::default()
        .eq_within_tols(&solution, &expected)
        .unwrap();
}

#[test]
fn refuses_a_nonpositive_curvature() {
    let matrix = indefinite();
    let rhs = right_hand_side();
    assert!(
        krylov()
            .solve_operator(apply(&matrix), Preconditioning::None, &rhs)
            .is_err()
    );
}

#[test]
fn minres_serves_the_indefinite_system() {
    let matrix = indefinite();
    let rhs = right_hand_side();
    let expected = matrix.clone().solve_lu(&rhs).unwrap();
    let solution = minres()
        .solve_operator(apply(&matrix), Preconditioning::None, &rhs)
        .unwrap();
    Assert::default()
        .eq_within_tols(&solution, &expected)
        .unwrap();
}
