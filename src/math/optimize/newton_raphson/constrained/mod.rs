use super::super::{
    super::{
        Hessian, Jacobian, LuDecomposition, Scalar, Solution, SquareMatrix, Tensor, Vector,
        sparse::SparseSolver,
    },
    Direct, OptimizationError, TrustRegion,
    trust_region::{
        TRUST_REGION_ETA, TRUST_REGION_MAX_REJECTIONS, TRUST_REGION_SHRINK, more_sorensen_dense,
        more_sorensen_sparse, update_radius,
    },
};
use super::constraint::{ConstraintMatrix, Eliminated, kkt_entry_sparse};
use crate::math::Norm;
use std::{fmt::Display, ops::Mul};

use super::NewtonRaphson;
use super::backtrack::{Backtrack, half_square, searched};
use super::convergence::{converged, limit_decrement};
use super::penalty::{backtrack_penalty, merit_slope, raise_penalty, violation};
use super::verify::not_minimum;

#[expect(clippy::too_many_arguments)]
pub(super) fn constrained<B, S, J, H, X, M>(
    newton_raphson: &NewtonRaphson<Direct, S>,
    search: B,
    mut function: impl FnMut(&X) -> Result<Scalar, String>,
    mut jacobian: impl FnMut(&X) -> Result<J, String>,
    mut hessian: impl FnMut(&X) -> Result<H, String>,
    mut update: impl FnMut(&X, &Vector, Scalar, bool) -> Result<(), String>,
    initial_guess: X,
    sparse: Option<SparseSolver>,
    eliminated: &[usize],
    constraint_matrix: M,
    constraint_rhs: Vector,
) -> Result<X, OptimizationError>
where
    B: Backtrack,
    S: Display,
    H: Hessian,
    J: Jacobian,
    X: Solution,
    M: ConstraintMatrix,
    for<'a> &'a M: Mul<&'a X, Output = Vector>,
{
    let mut penalty = 0.0 as Scalar;
    let num_variables = initial_guess.size();
    let eliminated = Eliminated::new(eliminated, num_variables);
    let mut applied = Vector::zero(num_variables);
    let num_constraints = constraint_rhs.len();
    let num_total = num_variables + num_constraints;
    let mut decrement = Vector::zero(num_total);
    let mut factorization = LuDecomposition::zero(if sparse.is_none() { num_total } else { 0 });
    let mut multipliers = Vector::zero(num_constraints);
    let mut residual = Vector::zero(num_total);
    let mut scales = None;
    let mut solution = initial_guess;
    let mut tangent = SquareMatrix::zero(if sparse.is_none() { num_total } else { 0 });
    if sparse.is_none() {
        constraint_matrix.for_each_nonzero(|i, j, constraint_matrix_ij| {
            tangent[i + num_variables][j] = -constraint_matrix_ij;
            tangent[j][i + num_variables] = -constraint_matrix_ij;
        });
    }
    eliminated.decouple(&mut tangent);
    let (mut radius, max_radius) =
        if let TrustRegion::Adaptive { radius, max_radius } = newton_raphson.trust_region {
            (radius, max_radius)
        } else {
            (0.0, 0.0)
        };
    let mut steps = 0;
    loop {
        eliminated
            .hold(jacobian(&solution)? - constraint_matrix.weighted(&multipliers))
            .fill_into_chained(
                &constraint_rhs - &constraint_matrix * &solution,
                &mut residual,
            );
        if converged(newton_raphson, &residual, num_variables, &mut scales) {
            if B::OBJECTIVE && sparse.is_none() {
                hessian(&solution)?.fill_into(&mut tangent);
                eliminated.decouple(&mut tangent);
                if tangent
                    .factorize_ldl()
                    .is_ok_and(|ldl| !ldl.is_minimum(num_constraints))
                {
                    return Err(not_minimum(newton_raphson, num_variables, num_constraints));
                }
            }
            if B::OBJECTIVE
                && let Some(ref solver) = sparse
            {
                let hess = hessian(&solution)?;
                let verdict = solver.is_minimum(
                    |i, j| {
                        kkt_entry_sparse(
                            &hess,
                            &constraint_matrix,
                            num_variables,
                            i,
                            j,
                            &eliminated,
                        )
                    },
                    num_constraints,
                );
                if verdict == Some(false) {
                    return Err(not_minimum(newton_raphson, num_variables, num_constraints));
                }
            }
            return Ok(solution);
        } else if steps == newton_raphson.max_steps {
            return Err(OptimizationError::MaximumStepsReached(
                newton_raphson.max_steps,
                format!("{newton_raphson:?}"),
            ));
        } else if B::OBJECTIVE
            && matches!(newton_raphson.trust_region, TrustRegion::Adaptive { .. })
        {
            let mut hess = Some(hessian(&solution)?);
            if sparse.is_none()
                && let Some(hess) = hess.take()
            {
                hess.fill_into(&mut tangent);
                eliminated.decouple(&mut tangent)
            }
            penalty = raise_penalty(
                penalty,
                multipliers.iter().zip(decrement.iter().skip(num_variables)),
            );
            let violated = penalty * violation(&constraint_matrix, &constraint_rhs, &solution);
            let value = function(&solution)? + violated;
            let mut gradient = Vector::zero(num_variables);
            eliminated
                .hold(jacobian(&solution)?)
                .fill_into(&mut gradient);
            let mut accepted = false;
            for _ in 0..TRUST_REGION_MAX_REJECTIONS {
                let quadratic = if let (Some(solver), Some(hess)) = (&sparse, &hess) {
                    decrement = more_sorensen_sparse(
                        solver,
                        |i, j| {
                            kkt_entry_sparse(
                                hess,
                                &constraint_matrix,
                                num_variables,
                                i,
                                j,
                                &eliminated,
                            )
                        },
                        &residual,
                        radius,
                        num_variables,
                    )?;
                    hess.quadratic_form(&decrement)
                } else {
                    decrement = more_sorensen_dense(&tangent, &residual, radius, num_variables);
                    tangent.quadratic_form_leading(&decrement, num_variables)
                };
                let predicted =
                    merit_slope(gradient.iter().zip(decrement.iter()), violated) - 0.5 * quadratic;
                applied
                    .iter_mut()
                    .zip(decrement.iter())
                    .for_each(|(applied_i, decrement_i)| *applied_i = *decrement_i);
                if predicted <= Scalar::EPSILON * value.abs() {
                    accepted = true;
                    break;
                }
                let mut trial = solution.clone();
                let mut trial_multipliers = multipliers.clone();
                trial.decrement_from_chained(&mut trial_multipliers, &decrement);
                let primal_norm =
                    Norm::Euclidean.over(decrement.iter().take(num_variables).copied());
                let trial_value = update(&solution, &applied, 1.0, false)
                    .and_then(|()| function(&trial))
                    .map(|value| {
                        value + penalty * violation(&constraint_matrix, &constraint_rhs, &trial)
                    });
                let Ok(value_trial) = trial_value else {
                    radius = TRUST_REGION_SHRINK * primal_norm;
                    continue;
                };
                let rho = (value - value_trial) / predicted;
                radius = update_radius(radius, max_radius, rho, primal_norm);
                if rho > TRUST_REGION_ETA {
                    accepted = true;
                    break;
                }
            }
            if !accepted {
                return Err(OptimizationError::MaximumStepsReached(
                    TRUST_REGION_MAX_REJECTIONS,
                    format!("{newton_raphson:?}"),
                ));
            }
            steps += 1;
            update(&solution, &applied, 1.0, true)?;
            solution.decrement_from_chained(&mut multipliers, &decrement);
            continue;
        } else if let Some(ref solver) = sparse {
            let hess = hessian(&solution)?;
            decrement = solver.solve(
                |i, j| {
                    kkt_entry_sparse(&hess, &constraint_matrix, num_variables, i, j, &eliminated)
                },
                &residual,
            )?;
        } else {
            hessian(&solution)?.fill_into(&mut tangent);
            eliminated.decouple(&mut tangent);
            tangent.factorize_lu_into(&mut factorization)?;
            factorization.solve_into(&residual, &mut decrement)
        }
        steps += 1;
        let fraction = limit_decrement(newton_raphson, &mut [(&mut decrement, num_variables)]);
        //
        // Only the variables are lent out, the multipliers chained onto the end
        // of the decrement being of another kind entirely.
        //
        applied
            .iter_mut()
            .zip(decrement.iter())
            .for_each(|(applied_i, decrement_i)| *applied_i = *decrement_i);
        let step_size = if !B::SEARCHES {
            1.0
        } else if !(B::OBJECTIVE && B::MERIT) {
            //
            // Without a merit of the objective to compare, a trial point is
            // judged by the residual it leaves, or by whether there is one.
            //
            let value = if B::MERIT {
                half_square(residual.iter().copied())
            } else {
                0.0
            };
            searched(
                newton_raphson,
                search.merit(
                    |trial_size| {
                        let mut trial = solution.clone();
                        let mut trial_multipliers = multipliers.clone();
                        trial.decrement_from_chained(
                            &mut trial_multipliers,
                            &(&decrement * trial_size),
                        );
                        update(&solution, &applied, trial_size, false)?;
                        let mut trial_residual = Vector::zero(num_total);
                        eliminated
                            .hold(
                                jacobian(&trial)? - constraint_matrix.weighted(&trial_multipliers),
                            )
                            .fill_into_chained(
                                &constraint_rhs - &constraint_matrix * &trial,
                                &mut trial_residual,
                            );
                        Ok(half_square(trial_residual.iter().copied()))
                    },
                    value,
                    0.0,
                    fraction,
                ),
            )?
        } else {
            penalty = raise_penalty(
                penalty,
                multipliers.iter().zip(decrement.iter().skip(num_variables)),
            );
            let violated = penalty * violation(&constraint_matrix, &constraint_rhs, &solution);
            let mut gradient = Vector::zero(num_variables);
            eliminated
                .hold(jacobian(&solution)?)
                .fill_into(&mut gradient);
            let slope = merit_slope(gradient.iter().zip(decrement.iter()), violated);
            update(&solution, &applied, 0.0, false)?;
            let value = function(&solution)? + violated;
            backtrack_penalty(
                newton_raphson,
                &search,
                |step| {
                    let mut trial = solution.clone();
                    let mut trial_multipliers = multipliers.clone();
                    trial.decrement_from_chained(&mut trial_multipliers, &(&decrement * step));
                    update(&solution, &applied, step, false)?;
                    Ok(function(&trial)?
                        + penalty * violation(&constraint_matrix, &constraint_rhs, &trial))
                },
                value,
                slope,
            )?
        };
        //
        // The increment is lent out whole, before it is applied and before it
        // is shortened, so that the eliminated variables take the same fraction
        // of their own direction as the retained ones take of theirs.
        //
        update(&solution, &applied, step_size, true)?;
        if step_size != 1.0 {
            decrement *= step_size
        }
        solution.decrement_from_chained(&mut multipliers, &decrement)
    }
}
