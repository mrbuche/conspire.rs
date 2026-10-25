use super::ConstraintMatrix;
use crate::math::optimize::{
    Direct, NewtonRaphson, NoSearch, OptimizationError, TrustRegion, no_objective,
    nonlinear::newton_raphson::{
        backtrack::{Objective, Residually},
        constrained::constrained,
    },
};
use crate::math::{
    Matrix, Scalar, SquareMatrix, Vector,
    assert::{Assert, AssertionError},
    sparse::{CscMatrix, SparseSolver},
};
use std::ops::Mul;

const SOLUTION: [Scalar; 3] = [0.8, 0.7, 0.7];

fn residual(x: &Vector) -> Result<Vector, String> {
    Ok([
        2.0 * x[0] - x[1] - 1.0,
        -x[0] + 2.0 * x[1] - x[2],
        -x[1] + 2.0 * x[2] - 1.0,
    ]
    .into())
}

fn tangent(_: &Vector) -> Result<SquareMatrix, String> {
    Ok([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]].into())
}

fn energy(x: &Vector) -> Result<Scalar, String> {
    Ok(0.5
        * (2.0 * x[0] * x[0] - 2.0 * x[0] * x[1] + 2.0 * x[1] * x[1] - 2.0 * x[1] * x[2]
            + 2.0 * x[2] * x[2])
        - x[0]
        - x[2])
}

fn dense() -> Matrix {
    let mut matrix = Matrix::zero(2, 3);
    matrix[0][2] = 1.0;
    matrix[1][0] = 1.0;
    matrix[1][1] = 1.0;
    matrix
}

fn sparse_matrix() -> CscMatrix {
    let mut matrix = CscMatrix::from_pattern(2, 3, vec![(0, 2), (1, 0), (1, 1)]);
    matrix.fill(|_, _| 1.0);
    matrix
}

fn right_hand_side() -> Vector {
    [0.7, 1.5].into()
}

fn solver() -> Option<SparseSolver> {
    Some(SparseSolver::from_pattern(
        5,
        vec![
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (3, 2),
            (2, 3),
            (4, 0),
            (4, 1),
            (0, 4),
            (1, 4),
        ],
        true,
    ))
}

fn rooted<M>(matrix: M, sparse: Option<SparseSolver>) -> Result<Vector, OptimizationError>
where
    M: ConstraintMatrix,
    for<'a> &'a M: Mul<&'a Vector, Output = Vector>,
{
    constrained(
        &NewtonRaphson::default(),
        Residually(&NoSearch),
        no_objective,
        residual,
        tangent,
        |_: &Vector, _: &Vector, _: Scalar, _: bool| Ok(()),
        Vector::zero(3),
        sparse,
        &[],
        matrix,
        right_hand_side(),
    )
}

fn minimized<M>(
    matrix: M,
    sparse: Option<SparseSolver>,
    trust_region: TrustRegion,
) -> Result<Vector, OptimizationError>
where
    M: ConstraintMatrix,
    for<'a> &'a M: Mul<&'a Vector, Output = Vector>,
{
    constrained(
        &NewtonRaphson {
            trust_region,
            linear_solver: Direct,
            ..Default::default()
        },
        Objective(&NoSearch),
        energy,
        residual,
        tangent,
        |_: &Vector, _: &Vector, _: Scalar, _: bool| Ok(()),
        Vector::zero(3),
        sparse,
        &[],
        matrix,
        right_hand_side(),
    )
}

#[test]
fn a_dense_matrix_finds_the_root() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(&rooted(dense(), None)?, &Vector::from(SOLUTION))
}

#[test]
fn a_sparse_matrix_finds_the_root() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(&rooted(sparse_matrix(), None)?, &Vector::from(SOLUTION))
}

#[test]
fn a_sparse_solver_finds_the_root_with_either_matrix() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(&rooted(dense(), solver())?, &Vector::from(SOLUTION))?;
    Assert::default().eq_within_tols(&rooted(sparse_matrix(), solver())?, &Vector::from(SOLUTION))
}

#[test]
fn either_matrix_finds_the_minimum() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        &minimized(dense(), None, TrustRegion::None)?,
        &Vector::from(SOLUTION),
    )?;
    Assert::default().eq_within_tols(
        &minimized(sparse_matrix(), None, TrustRegion::None)?,
        &Vector::from(SOLUTION),
    )
}

#[test]
fn a_sparse_solver_finds_the_minimum_with_either_matrix() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        &minimized(dense(), solver(), TrustRegion::None)?,
        &Vector::from(SOLUTION),
    )?;
    Assert::default().eq_within_tols(
        &minimized(sparse_matrix(), solver(), TrustRegion::None)?,
        &Vector::from(SOLUTION),
    )
}

#[test]
fn the_trust_region_reads_either_matrix_entry_by_entry() -> Result<(), AssertionError> {
    let adaptive = || TrustRegion::Adaptive {
        radius: 1.0,
        max_radius: 10.0,
    };
    Assert::default().eq_within_tols(
        &minimized(dense(), solver(), adaptive())?,
        &Vector::from(SOLUTION),
    )?;
    Assert::default().eq_within_tols(
        &minimized(sparse_matrix(), solver(), adaptive())?,
        &Vector::from(SOLUTION),
    )
}
