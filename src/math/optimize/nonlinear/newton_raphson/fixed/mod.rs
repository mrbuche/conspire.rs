use crate::math::{
    Erase, Jacobian, Scalar, Solution, Tensor, Vector,
    optimize::{OptimizationError, StepSize},
};
use crate::units::UnitRatio;
use std::{fmt::Display, ops::Mul};

use super::NewtonRaphson;
use super::backtrack::{Backtrack, half_square, residual_step, searched};
use super::convergence::limit_decrement;
use super::verify::not_minimum;

#[expect(clippy::too_many_arguments)]
pub(super) fn constrained_fixed<B, L, S, J, H, X, E>(
    newton_raphson: &NewtonRaphson<L, S>,
    search: B,
    mut function: impl FnMut(&X) -> Result<Scalar, String>,
    mut jacobian: impl FnMut(&X) -> Result<J, String>,
    mut hessian: impl FnMut(&X) -> Result<H, String>,
    mut update: impl FnMut(&X, &Vector, Scalar, bool) -> Result<(), String>,
    initial_guess: X,
    mut solve: impl FnMut(H, &[bool], &[usize], &Vector, &mut Vector) -> Result<(), OptimizationError>,
    mut verify: Option<impl FnMut(H, &[bool], &[usize]) -> Option<bool>>,
    indices: Vec<usize>,
) -> Result<X, OptimizationError>
where
    B: Backtrack,
    S: Display,
    J: Jacobian,
    J: Erase<Erased = E>,
    X: Erase<Erased = E> + Solution,
    E: Tensor,
    <X as Tensor>::Unit: UnitRatio,
    for<'a> &'a X: Mul<StepSize<X, X>, Output = X>,
{
    let mut applied = Vector::zero(initial_guess.size());
    let mut retained = vec![true; initial_guess.size()];
    indices.iter().for_each(|&index| retained[index] = false);
    let unmap: Vec<usize> = retained
        .iter()
        .enumerate()
        .filter_map(|(index, &keep)| keep.then_some(index))
        .collect();
    let mut decrement = Vector::zero(unmap.len());
    let mut residual;
    let mut solution = initial_guess;
    let mut step_size;
    let mut steps = 0;
    loop {
        residual = jacobian(&solution)?.retain_from(&retained);
        if newton_raphson.error_norm.apply(&residual) < newton_raphson.abs_tol.residual() {
            if let Some(verify) = verify.as_mut()
                && verify(hessian(&solution)?, &retained, &unmap) == Some(false)
            {
                return Err(not_minimum(newton_raphson, unmap.len(), 0));
            }
            return Ok(solution);
        } else if steps == newton_raphson.max_steps {
            return Err(OptimizationError::MaximumStepsReached(
                newton_raphson.max_steps,
                format!("{newton_raphson:?}"),
            ));
        } else {
            solve(
                hessian(&solution)?,
                &retained,
                &unmap,
                &residual,
                &mut decrement,
            )?
        }
        steps += 1;
        let fraction = limit_decrement(newton_raphson, &mut [(&mut decrement, unmap.len())]);
        applied.iter_mut().for_each(|entry| *entry = 0.0);
        unmap
            .iter()
            .zip(decrement.iter())
            .for_each(|(&index, decrement_a)| applied[index] = *decrement_a);
        step_size = if !B::SEARCHES {
            1.0
        } else if !(B::OBJECTIVE && B::MERIT) {
            residual_step(
                newton_raphson,
                &search,
                || half_square(residual.iter().copied()),
                |trial_size| {
                    let mut trial = solution.clone();
                    trial.decrement_from_retained(&retained, &(&decrement * trial_size));
                    update(&solution, &applied, trial_size, false)?;
                    Ok(half_square(
                        jacobian(&trial)?.retain_from(&retained).iter().copied(),
                    ))
                },
                fraction,
            )?
        } else {
            let jac = jacobian(&solution)?;
            let mut decrement_full = solution.clone();
            decrement_full *= 0.0;
            decrement_full.decrement_from_retained(&retained, &decrement);
            decrement_full *= -1.0;
            searched(
                newton_raphson,
                search.gradient(
                    |trial: &X, step: Scalar| {
                        update(&solution, &applied, step, false)?;
                        function(trial)
                    },
                    &mut jacobian,
                    &solution,
                    &jac,
                    &decrement_full,
                    1.0,
                ),
            )?
        };
        update(&solution, &applied, step_size, true)?;
        if step_size != 1.0 {
            decrement *= step_size
        }
        solution.decrement_from_retained(&retained, &decrement)
    }
}
