use crate::math::{
    Hessian, Jacobian, Scalar, Solution, SquareMatrix, Tensor, Vector,
    optimize::{Direct, OptimizationError, StepSize, TrustRegion},
    sparse::SparseSolver,
};
use crate::units::UnitRatio;
use std::{
    fmt::Display,
    ops::{Div, Mul},
};

use super::NewtonRaphson;
use super::backtrack::{Backtrack, Gradient, half_square_norm, residual_step, searched};
use super::verify::not_minimum;

pub(super) fn unconstrained<B, S, J, H, X>(
    newton_raphson: &NewtonRaphson<Direct, S>,
    search: B,
    mut function: impl FnMut(&X) -> Result<Scalar, String>,
    mut jacobian: impl FnMut(&X) -> Result<J, String>,
    mut hessian: impl FnMut(&X) -> Result<H, String>,
    initial_guess: X,
    sparse: Option<SparseSolver>,
) -> Result<X, OptimizationError>
where
    B: Backtrack + Gradient<X, J, X>,
    S: Display,
    H: Hessian,
    J: Jacobian,
    for<'a> &'a J: Div<H, Output = X>,
    X: Solution,
    <X as Tensor>::Unit: UnitRatio,
    for<'a> &'a X: Mul<StepSize<X, X>, Output = X> + Mul<Scalar, Output = X>,
{
    let mut decrement;
    let mut flattened = Vector::zero(if sparse.is_none() {
        0
    } else {
        initial_guess.size()
    });
    let mut residual;
    let mut solution = initial_guess;
    let mut step_size;
    let mut steps = 0;
    loop {
        residual = jacobian(&solution)?;
        if newton_raphson.error_norm.apply(&residual) < newton_raphson.abs_tol.residual() {
            if B::OBJECTIVE {
                let hess = hessian(&solution)?;
                let verdict = if let Some(ref solver) = sparse {
                    solver.is_minimum(|i, j| hess.entry(i, j), 0)
                } else {
                    let mut square = SquareMatrix::zero(solution.size());
                    hess.fill_into(&mut square);
                    square.factorize_ldl().ok().map(|ldl| ldl.is_minimum(0))
                };
                if verdict == Some(false) {
                    return Err(not_minimum(newton_raphson, solution.size(), 0));
                }
            }
            return Ok(solution);
        } else if steps == newton_raphson.max_steps {
            return Err(OptimizationError::MaximumStepsReached(
                newton_raphson.max_steps,
                format!("{newton_raphson:?}"),
            ));
        } else {
            steps += 1;
            decrement = if let Some(ref solver) = sparse {
                let hess = hessian(&solution)?;
                residual.fill_into(&mut flattened);
                X::from(solver.solve(|i, j| hess.entry(i, j), &flattened)?)
            } else {
                &residual / hessian(&solution)?
            };
            let mut fraction = 1.0;
            if let TrustRegion::Fixed { radius, norm } = newton_raphson.trust_region {
                let size = norm.measure(&decrement);
                if size > radius {
                    decrement *= radius / size;
                    fraction = radius / size
                }
            }
            step_size = if !B::SEARCHES {
                1.0
            } else if B::OBJECTIVE {
                searched(
                    newton_raphson,
                    search.gradient(
                        |trial: &X, _: Scalar| function(trial),
                        &mut jacobian,
                        &solution,
                        &residual,
                        &decrement,
                        1.0,
                    ),
                )?
            } else {
                residual_step(
                    newton_raphson,
                    &search,
                    || half_square_norm(&residual),
                    |trial_size| {
                        let mut trial = solution.clone();
                        trial -= &decrement * trial_size;
                        Ok(half_square_norm(&jacobian(&trial)?))
                    },
                    fraction,
                )?
            };
            if step_size != 1.0 {
                decrement *= step_size
            }
            solution -= decrement
        }
    }
}
