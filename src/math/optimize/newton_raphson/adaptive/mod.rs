use super::super::{
    super::{
        Hessian, Jacobian, Scalar, Solution, SquareMatrix, Tensor, Vector, sparse::SparseSolver,
    },
    Direct, OptimizationError,
    trust_region::{
        TRUST_REGION_ETA, TRUST_REGION_MAX_REJECTIONS, TRUST_REGION_SHRINK, more_sorensen_dense,
        more_sorensen_sparse, update_radius,
    },
};
use super::NewtonRaphson;
use super::constraint::{ConstraintMatrix, Eliminated, kkt_entry_sparse};
use super::penalty::{merit_slope, raise_penalty, violation};
use crate::math::Norm;
use std::{fmt::Display, ops::Mul};

/// One step of a constrained minimization within an adaptive trust region.
///
/// The step is the one that the region around the solution and the predicted
/// decrease of the penalty merit agree on, shrunk or grown until the merit
/// falls by enough of what was predicted. The solution and its multipliers
/// are then moved, and the update lent out whole, as the line search does.
#[expect(clippy::too_many_arguments)]
pub(super) fn adaptive_step<S, J, H, X, M>(
    newton_raphson: &NewtonRaphson<Direct, S>,
    function: &mut impl FnMut(&X) -> Result<Scalar, String>,
    jacobian: &mut impl FnMut(&X) -> Result<J, String>,
    hessian: &mut impl FnMut(&X) -> Result<H, String>,
    update: &mut impl FnMut(&X, &Vector, Scalar, bool) -> Result<(), String>,
    solution: &mut X,
    multipliers: &mut Vector,
    decrement: &mut Vector,
    applied: &mut Vector,
    tangent: &mut SquareMatrix,
    penalty: &mut Scalar,
    radius: &mut Scalar,
    max_radius: Scalar,
    residual: &Vector,
    eliminated: &Eliminated,
    constraint_matrix: &M,
    constraint_rhs: &Vector,
    sparse: &Option<SparseSolver>,
) -> Result<(), OptimizationError>
where
    S: Display,
    H: Hessian,
    J: Jacobian,
    X: Solution,
    M: ConstraintMatrix,
    for<'a> &'a M: Mul<&'a X, Output = Vector>,
{
    let num_variables = solution.size();
    let mut hess = Some(hessian(&*solution)?);
    if sparse.is_none()
        && let Some(hess) = hess.take()
    {
        hess.fill_into(tangent);
        eliminated.decouple(tangent)
    }
    *penalty = raise_penalty(
        *penalty,
        multipliers.iter().zip(decrement.iter().skip(num_variables)),
    );
    let violated = *penalty * violation(constraint_matrix, constraint_rhs, &*solution);
    let value = function(&*solution)? + violated;
    let mut gradient = Vector::zero(num_variables);
    eliminated
        .hold(jacobian(&*solution)?)
        .fill_into(&mut gradient);
    let mut accepted = false;
    for _ in 0..TRUST_REGION_MAX_REJECTIONS {
        let quadratic = if let (Some(solver), Some(hess)) = (sparse, &hess) {
            *decrement = more_sorensen_sparse(
                solver,
                |i, j| kkt_entry_sparse(hess, constraint_matrix, num_variables, i, j, eliminated),
                residual,
                *radius,
                num_variables,
            )?;
            hess.quadratic_form(decrement)
        } else {
            *decrement = more_sorensen_dense(tangent, residual, *radius, num_variables);
            tangent.quadratic_form_leading(decrement, num_variables)
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
        trial.decrement_from_chained(&mut trial_multipliers, decrement);
        let primal_norm = Norm::Euclidean.over(decrement.iter().take(num_variables).copied());
        let trial_value = update(&*solution, applied, 1.0, false)
            .and_then(|()| function(&trial))
            .map(|value| value + *penalty * violation(constraint_matrix, constraint_rhs, &trial));
        let Ok(value_trial) = trial_value else {
            *radius = TRUST_REGION_SHRINK * primal_norm;
            continue;
        };
        let rho = (value - value_trial) / predicted;
        *radius = update_radius(*radius, max_radius, rho, primal_norm);
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
    update(&*solution, applied, 1.0, true)?;
    solution.decrement_from_chained(multipliers, decrement);
    Ok(())
}
