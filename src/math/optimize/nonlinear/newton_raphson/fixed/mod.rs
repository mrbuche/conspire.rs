use crate::math::{
    Jacobian, Scalar, Solution, Tensor, Vector,
    optimize::{OptimizationError, StepSize},
};
use crate::units::UnitRatio;
use std::{fmt::Display, ops::Mul};

use super::NewtonRaphson;
use super::backtrack::{Backtrack, Gradient, half_square_norm, residual_step, searched};
use super::convergence::shorten;
use super::verify::not_minimum;

#[expect(clippy::too_many_arguments)]
pub(super) fn constrained_fixed<B, L, S, J, H, X>(
    newton_raphson: &NewtonRaphson<L, S>,
    search: B,
    mut function: impl FnMut(&X) -> Result<Scalar, String>,
    mut jacobian: impl FnMut(&X) -> Result<J, String>,
    mut hessian: impl FnMut(&X) -> Result<H, String>,
    mut update: impl FnMut(&X, &Vector, Scalar, bool) -> Result<(), String>,
    initial_guess: X,
    mut solve: impl FnMut(H, &[usize], &J) -> Result<X, OptimizationError>,
    mut verify: Option<impl FnMut(H, &[bool], &[usize]) -> Option<bool>>,
    indices: Vec<usize>,
) -> Result<X, OptimizationError>
where
    B: Backtrack + Gradient<X, J, X>,
    S: Display,
    J: Jacobian,
    X: Jacobian + Solution,
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
    let mut residual;
    let mut solution = initial_guess;
    let mut step_size;
    let mut steps = 0;
    loop {
        residual = jacobian(&solution)?;
        residual.zero_out(&indices);
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
        }
        let mut decrement = solve(hessian(&solution)?, &indices, &residual)?;
        steps += 1;
        let fraction = shorten(newton_raphson, &mut decrement);
        decrement.fill_into(&mut applied);
        step_size = if !B::SEARCHES {
            1.0
        } else if !(B::OBJECTIVE && B::MERIT) {
            residual_step(
                newton_raphson,
                &search,
                || half_square_norm(&residual),
                |trial_size| {
                    let mut trial = solution.clone();
                    trial -= decrement.clone() * trial_size;
                    update(&solution, &applied, trial_size, false)?;
                    let mut leaves = jacobian(&trial)?;
                    leaves.zero_out(&indices);
                    Ok(half_square_norm(&leaves))
                },
                fraction,
            )?
        } else {
            let jac = jacobian(&solution)?;
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
                    &decrement,
                    1.0,
                ),
            )?
        };
        update(&solution, &applied, step_size, true)?;
        if step_size != 1.0 {
            decrement *= step_size
        }
        solution -= decrement
    }
}
