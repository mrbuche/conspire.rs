use super::{Krylov, KrylovError, Preconditioner};
use crate::math::{
    Hessian, SquareMatrix, Vector,
    assert::{Assert, AssertionError},
};

/// Tight enough that the residual left behind is below what the solution is
/// then compared to, an iterative solve being only as exact as it is asked to be.
fn krylov(preconditioner: Preconditioner) -> Krylov {
    Krylov {
        preconditioner,
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

fn right_hand_side() -> Vector {
    Vector::from([1.0, 2.0, 3.0])
}

/// Whichever answer the direct solve gives is the one to match.
fn direct(matrix: &SquareMatrix, right_hand_side: &Vector) -> Vector {
    matrix.clone().solve_lu(right_hand_side).unwrap()
}

/// Diagonal, positive definite, and every entry a decade apart from the next.
///
/// Conjugate gradients walks one direction per distinct eigenvalue, so this is
/// the worst case for going unpreconditioned and the best case for dividing by
/// the diagonal, which here is the whole of the matrix.
fn spread(size: usize) -> SquareMatrix {
    let mut matrix = SquareMatrix::zero(size);
    (0..size).for_each(|i| matrix[i][i] = 10.0_f64.powi(i as i32 % 12));
    matrix
}

fn ones(size: usize) -> Vector {
    (0..size).map(|_| 1.0).collect()
}

#[test]
fn matches_the_direct_solve() -> Result<(), AssertionError> {
    for preconditioner in [Preconditioner::Jacobi, Preconditioner::None] {
        Assert::default().eq_within_tols(
            &krylov(preconditioner).conjugate_gradients(&coupled(), &right_hand_side())?,
            &direct(&coupled(), &right_hand_side()),
        )?
    }
    Ok(())
}

/// Dividing by the diagonal of a diagonal matrix leaves the identity, which is
/// solved in a single direction however many distinct eigenvalues it started
/// with. Without it there is a direction to walk for each of them.
#[test]
fn jacobi_needs_one_iteration_where_nothing_needs_many() {
    let converges = |preconditioner, max_steps| {
        Krylov {
            max_steps,
            preconditioner,
            rel_tol: 1e-12,
        }
        .conjugate_gradients(&spread(24), &ones(24))
        .is_ok()
    };
    assert!(converges(Preconditioner::Jacobi, 1));
    assert!(!converges(Preconditioner::None, 1));
    assert!(!converges(Preconditioner::None, 8))
}

/// A negative eigenvalue is not something conjugate gradients can descend, and
/// the curvature along a direction is where that shows up.
#[test]
fn indefinite_is_refused() {
    let mut matrix = SquareMatrix::zero(2);
    matrix[0][0] = 1.0;
    matrix[1][1] = -1.0;
    assert!(matches!(
        krylov(Preconditioner::None)
            .conjugate_gradients(&matrix, &Vector::from([1.0, 1.0]))
            .unwrap_err(),
        KrylovError::NotPositiveDefinite(..)
    ))
}

#[test]
fn zero_right_hand_side_is_already_solved() -> Result<(), AssertionError> {
    Assert::default().zero_within_tols(
        &krylov(Preconditioner::Jacobi).conjugate_gradients(&coupled(), &Vector::zero(3))?,
    )
}

/// How far it got is reported, and that need not be anywhere nearer than where
/// it started.
///
/// What descends without fail is the error measured against the tangent, not the
/// residual measured against itself, so the number reported here is above one
/// and is not a sign of anything having gone wrong.
#[test]
fn too_few_iterations_reports_how_far_it_got() {
    let krylov = Krylov {
        max_steps: 2,
        preconditioner: Preconditioner::None,
        rel_tol: 1e-12,
    };
    match krylov
        .conjugate_gradients(&spread(24), &ones(24))
        .unwrap_err()
    {
        KrylovError::MaximumStepsReached(steps, relative) => {
            assert_eq!(steps, 2);
            assert!(relative.is_finite() && relative > 0.0)
        }
        error => panic!("Expected the iterations to run out, got {error}."),
    }
}

/// Restricting is scatter and gather around the whole product, so it has to
/// agree with striking the same rows and columns out of the matrix.
#[test]
fn retained_matches_the_restricted_direct_solve() -> Result<(), AssertionError> {
    let retained = [true, false, true];
    let unmap: Vec<usize> = (0..3).filter(|&i| retained[i]).collect();
    let reduced = coupled().retain_from(&retained);
    let right_hand_side = Vector::from([1.0, 3.0]);
    Assert::default().eq_within_tols(
        &krylov(Preconditioner::Jacobi).conjugate_gradients_retained(
            &coupled(),
            &unmap,
            3,
            &right_hand_side,
        )?,
        &direct(&reduced, &right_hand_side),
    )
}

/// A tangent reaching off the diagonal into a struck row must not feel it, the
/// struck variable standing at zero rather than at whatever it was.
#[test]
fn retained_ignores_what_was_struck_out() -> Result<(), AssertionError> {
    let mut matrix = SquareMatrix::zero(2);
    matrix[0][0] = 2.0;
    matrix[1][1] = 3.0;
    matrix[0][1] = 5.0;
    matrix[1][0] = 5.0;
    Assert::default().eq_within_tols(
        &krylov(Preconditioner::None).conjugate_gradients_retained(
            &matrix,
            &[0],
            2,
            &Vector::from([4.0]),
        )?,
        &Vector::from([2.0]),
    )
}

#[test]
fn display() {
    let _ = format!("{}", KrylovError::NotPositiveDefinite(-1.0));
    let _ = format!("{}", KrylovError::MaximumStepsReached(1, 0.5));
    let _ = format!("{:?}", Krylov::default());
}
