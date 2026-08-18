use super::{ACCEPTABLE, Krylov, KrylovError, KrylovMethod, Preconditioner, Preconditioning};
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

fn minres(preconditioner: Preconditioner) -> Krylov {
    Krylov {
        method: KrylovMethod::Minres,
        preconditioner,
        rel_tol: 1e-15,
        ..Default::default()
    }
}

/// Symmetric, and indefinite: the leading block is positive and the trailing one
/// negative, which is the shape a constraint gives a system.
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
            &krylov(preconditioner).solve(&coupled(), &right_hand_side())?,
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
            ..Default::default()
        }
        .solve(&spread(24), &ones(24))
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
            .solve(&matrix, &Vector::from([1.0, 1.0]))
            .unwrap_err(),
        KrylovError::NotPositiveDefinite(..)
    ))
}

#[test]
fn zero_right_hand_side_is_already_solved() -> Result<(), AssertionError> {
    Assert::default()
        .zero_within_tols(&krylov(Preconditioner::Jacobi).solve(&coupled(), &Vector::zero(3))?)
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
        ..Default::default()
    };
    match krylov.solve(&spread(24), &ones(24)).unwrap_err() {
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
        &krylov(Preconditioner::Jacobi).solve_retained(&coupled(), &unmap, 3, &right_hand_side)?,
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
        &krylov(Preconditioner::None).solve_retained(&matrix, &[0], 2, &Vector::from([4.0]))?,
        &Vector::from([2.0]),
    )
}

#[test]
fn display() {
    let _ = format!("{}", KrylovError::NotPositiveDefinite(-1.0));
    let _ = format!("{}", KrylovError::MaximumStepsReached(1, 0.5));
    let _ = format!("{:?}", Krylov::default());
}

/// The minimal residual method asks only for symmetry, so it reaches the same
/// answer conjugate gradients does where both apply, and an answer where only it
/// does.
mod minimal_residual {
    use super::*;

    #[test]
    fn matches_the_direct_solve_on_a_definite_system() -> Result<(), AssertionError> {
        for preconditioner in [Preconditioner::Jacobi, Preconditioner::None] {
            Assert::default().eq_within_tols(
                &minres(preconditioner).solve(&coupled(), &right_hand_side())?,
                &direct(&coupled(), &right_hand_side()),
            )?
        }
        Ok(())
    }

    /// The system with a zero on its diagonal is what a constraint row looks
    /// like, and is the one the preconditioner has to leave alone rather than
    /// divide by.
    #[test]
    fn matches_the_direct_solve_on_an_indefinite_system() -> Result<(), AssertionError> {
        for preconditioner in [Preconditioner::Jacobi, Preconditioner::None] {
            Assert::default().eq_within_tols(
                &minres(preconditioner).solve(&indefinite(), &right_hand_side())?,
                &direct(&indefinite(), &right_hand_side()),
            )?
        }
        Ok(())
    }

    /// The same system, told apart by which walk was asked for.
    #[test]
    fn succeeds_where_conjugate_gradients_refuses() {
        assert!(
            krylov(Preconditioner::None)
                .solve(&indefinite(), &right_hand_side())
                .is_err()
        );
        assert!(
            minres(Preconditioner::None)
                .solve(&indefinite(), &right_hand_side())
                .is_ok()
        )
    }

    #[test]
    fn zero_right_hand_side_is_already_solved() -> Result<(), AssertionError> {
        Assert::default().zero_within_tols(
            &minres(Preconditioner::Jacobi).solve(&indefinite(), &Vector::zero(3))?,
        )
    }

    #[test]
    fn retained_matches_the_restricted_direct_solve() -> Result<(), AssertionError> {
        let retained = [true, false, true];
        let unmap: Vec<usize> = (0..3).filter(|&i| retained[i]).collect();
        let reduced = indefinite().retain_from(&retained);
        let right_hand_side = Vector::from([1.0, 3.0]);
        Assert::default().eq_within_tols(
            &minres(Preconditioner::Jacobi).solve_retained(
                &indefinite(),
                &unmap,
                3,
                &right_hand_side,
            )?,
            &direct(&reduced, &right_hand_side),
        )
    }

    /// Small systems leave the rotation bookkeeping too little to do to be
    /// wrong in, so this is a larger one, symmetric and indefinite, built the
    /// same way every run.
    #[test]
    fn matches_the_direct_solve_on_a_larger_indefinite_system() -> Result<(), AssertionError> {
        let size = 30;
        let mut state = 12345_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1_u64 << 31) as f64) - 1.0
        };
        let mut matrix = SquareMatrix::zero(size);
        (0..size).for_each(|i| {
            (i..size).for_each(|j| {
                let entry = next();
                matrix[i][j] = entry;
                matrix[j][i] = entry
            })
        });
        let right_hand_side: Vector = (0..size).map(|_| next()).collect();
        Assert::default().eq_within_tols(
            &minres(Preconditioner::Jacobi).solve(&matrix, &right_hand_side)?,
            &direct(&matrix, &right_hand_side),
        )
    }

    #[test]
    fn too_few_iterations_reports_how_far_it_got() {
        let krylov = Krylov {
            max_steps: 1,
            method: KrylovMethod::Minres,
            preconditioner: Preconditioner::None,
            rel_tol: 1e-12,
        };
        assert!(matches!(
            krylov.solve(&spread(24), &ones(24)).unwrap_err(),
            KrylovError::MaximumStepsReached(1, _)
        ))
    }
}

/// A system large enough that the walk cannot simply exhaust the subspace.
///
/// A small system is solved in as many iterations as it has rows whatever it is
/// put through first, so nothing done to it before the walk can be told from
/// nothing at all. These are the tests that can tell.
mod at_size {
    use super::*;
    use crate::math::Tensor;

    /// A saddle-point system: a stiff, coupled, indefinite block with a
    /// constraint bordering it, which is the shape the constrained solves
    /// actually meet.
    fn saddle(variables: usize, constraints: usize) -> SquareMatrix {
        let size = variables + constraints;
        let mut matrix = SquareMatrix::zero(size);
        (0..variables).for_each(|i| {
            //
            // A diagonal that turns over every fourth row makes the leading
            // block indefinite in itself, which is what a tangent away from a
            // minimum is and what the magnitudes of an incomplete factorization
            // are for. Every diagonal entry outweighs the row it sits in, so
            // that the block is indefinite without being singular.
            //
            matrix[i][i] = if i % 4 == 3 {
                -12.0
            } else {
                12.0 + (i % 7) as f64
            };
            if i > 0 {
                matrix[i][i - 1] = -1.0;
                matrix[i - 1][i] = -1.0
            }
            if i >= 5 {
                matrix[i][i - 5] = -2.0;
                matrix[i - 5][i] = -2.0
            }
        });
        //
        // Each constraint reaches two variables of its own, so the constraints
        // are independent of one another and the system as a whole is not
        // singular.
        //
        (0..constraints).for_each(|a| {
            [3 * a, 3 * a + 1].into_iter().for_each(|j| {
                matrix[variables + a][j] = -1.0;
                matrix[j][variables + a] = -1.0
            })
        });
        matrix
    }

    fn load(size: usize) -> Vector {
        (0..size).map(|i| ((i % 13) as f64 - 6.0) / 7.0).collect()
    }

    /// Whatever the residual is put through first, the answer at the end is the
    /// system's own — a preconditioner changes how far the walk has to go, never
    /// where it arrives.
    #[test]
    fn every_preconditioner_reaches_the_direct_answer() -> Result<(), AssertionError> {
        let (variables, constraints) = (240, 20);
        let matrix = saddle(variables, constraints);
        let right_hand_side = load(variables + constraints);
        let expected = direct(&matrix, &right_hand_side);
        for preconditioner in [
            Preconditioner::None,
            Preconditioner::Jacobi,
            Preconditioner::IncompleteLdl {
                fill: 0,
                threshold: 0.0,
            },
            Preconditioner::IncompleteLdl {
                fill: 8,
                threshold: 0.0,
            },
        ] {
            let solution = minres(preconditioner).solve(&matrix, &right_hand_side)?;
            //
            // Measured against the system rather than against the answer: what
            // the walk promises is a short residual, and on an ill-conditioned
            // system a short residual still leaves the answer some way off.
            //
            let residual = (matrix.clone() * &solution - right_hand_side.clone())
                .norm()
                .value();
            assert!(
                residual < 1e-8 * right_hand_side.norm().value(),
                "{preconditioner:?} left a residual of {residual:?}"
            );
            Assert {
                rel_tol: 1e-6,
                ..Default::default()
            }
            .eq_within_tols(&solution, &expected)?
        }
        Ok(())
    }

    /// The whole point of putting the residual through anything: the walk that
    /// does gets there in fewer iterations than the walk that does not.
    #[test]
    fn preconditioning_shortens_the_walk() -> Result<(), AssertionError> {
        let (variables, constraints) = (240, 20);
        let matrix = saddle(variables, constraints);
        let right_hand_side = load(variables + constraints);
        let steps = |preconditioner| {
            (1..=variables + constraints).find(|&max_steps| {
                Krylov {
                    max_steps,
                    method: KrylovMethod::Minres,
                    preconditioner,
                    rel_tol: 1e-10,
                }
                .solve(&matrix, &right_hand_side)
                .is_ok()
            })
        };
        let bare = steps(Preconditioner::None).expect("unpreconditioned never converged");
        let factored = steps(Preconditioner::IncompleteLdl {
            fill: 8,
            threshold: 0.0,
        })
        .expect("preconditioned never converged");
        assert!(
            factored < bare,
            "preconditioned took {factored} against {bare}"
        );
        Ok(())
    }
}

/// What the rotations report is the residual seen through the preconditioner,
/// and a badly conditioned preconditioner makes that a different number from the
/// residual — small when the residual is not. A walk that took the report for
/// the residual would stop early and hand back an answer it had not reached,
/// saying it had.
///
/// So the walk may refuse this system or it may answer it, but the one thing it
/// may not do is call a long residual a short one.
#[test]
fn a_misleading_preconditioner_does_not_pass_for_a_short_residual() {
    use crate::math::Tensor;
    let size = 80;
    let mut matrix = SquareMatrix::zero(size);
    (0..size).for_each(|i| {
        matrix[i][i] = if i % 3 == 2 { -4.0 } else { 9.0 };
        if i > 0 {
            matrix[i][i - 1] = -1.0;
            matrix[i - 1][i] = -1.0
        }
    });
    //
    // The load sits only where the preconditioner weighs heaviest. The walk
    // shortens the residual measured through the preconditioner, so it spends
    // itself on those coordinates and leaves the residual on the ones weighed
    // lightest — where it is long, and where the measurement barely registers
    // it. That is the gap between the two, opened on purpose.
    //
    let right_hand_side: Vector = (0..size)
        .map(|i| {
            if i % 2 == 0 {
                (i % 5) as f64 + 1.0
            } else {
                0.0
            }
        })
        .collect();
    //
    // Positive definite, as the walk requires, and spanning twenty-four decades.
    //
    let preconditioning = Preconditioning::Diagonal(
        (0..size)
            .map(|i| 10.0_f64.powi(if i % 2 == 0 { -12 } else { 12 }))
            .collect(),
    );
    let krylov = Krylov {
        max_steps: 400,
        method: KrylovMethod::Minres,
        preconditioner: Preconditioner::Jacobi,
        rel_tol: 1e-8,
    };
    if let Ok(solution) =
        krylov.solve_operator(|v| matrix.times(v), preconditioning, &right_hand_side)
    {
        let residual = (matrix.clone() * &solution - right_hand_side.clone())
            .norm()
            .value()
            / right_hand_side.norm().value();
        assert!(
            residual <= ACCEPTABLE,
            "answered with a solution leaving a relative residual of {residual:?}"
        )
    }
}
