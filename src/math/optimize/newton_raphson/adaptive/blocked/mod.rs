use super::super::super::{
    super::{Jacobian, Matrix, Scalar, Solution, SquareMatrix, Tensor, Vector, sparse::CscMatrix},
    OptimizationError,
    trust_region::{
        TRUST_REGION_ETA, TRUST_REGION_MAX_REJECTIONS, TRUST_REGION_SHRINK, more_sorensen_dense,
        update_radius,
    },
};
use super::super::NewtonRaphson;
use super::super::penalty::{merit_slope, raise_penalty, violation};
use crate::math::Norm;
use std::{fmt::Display, ops::Mul};

/// One step of a minimization in global and local variables within an
/// adaptive trust region, once the local variables are eliminated.
///
/// The model the region bounds is the reduced one, whose gradient and tangent
/// have the local variables solved out, together with the decrease the local
/// variables buy on their own. The step is shrunk or grown until the penalty
/// merit falls by enough of what was predicted, then both blocks are moved.
#[expect(clippy::too_many_arguments)]
pub(in super::super) fn adaptive_step_blocked<L, S, U, V, Ru, Rv>(
    newton_raphson: &NewtonRaphson<L, S>,
    function: &mut impl FnMut(&U, &V) -> Result<Scalar, String>,
    residual_global: &mut impl FnMut(&U, &V) -> Result<Ru, String>,
    residual_local: &mut impl FnMut(&U, &V) -> Result<Rv, String>,
    global: &mut U,
    local: &mut V,
    multipliers_global: &mut Vector,
    multipliers_local: &mut Vector,
    decrement_outer: &mut Vector,
    decrement_inner: &mut Vector,
    penalty: &mut Scalar,
    radius: &mut Scalar,
    max_radius: Scalar,
    tangent_outer: &SquareMatrix,
    update_outer: &Vector,
    coupling_global: &Matrix,
    eliminated: &[Vector],
    (constraint_matrix_global, constraint_rhs_global): (&CscMatrix, &Vector),
    (constraint_matrix_local, constraint_rhs_local): (&CscMatrix, &Vector),
) -> Result<(), OptimizationError>
where
    S: Display,
    U: Solution,
    V: Solution,
    Ru: Jacobian,
    Rv: Jacobian,
    for<'a> &'a CscMatrix: Mul<&'a U, Output = Vector> + Mul<&'a V, Output = Vector>,
{
    let num_global = global.size();
    let num_local = local.size();
    let decrement_local = decrement_inner.clone();
    let mut gradient_global = Vector::zero(num_global);
    let mut gradient_local = Vector::zero(num_local);
    residual_global(&*global, &*local)?.fill_into(&mut gradient_global);
    residual_local(&*global, &*local)?.fill_into(&mut gradient_local);
    //
    // What the reduced model sees of the gradient, and the decrease
    // the local variables buy on their own before the global ones
    // move at all.
    //
    let reduced_gradient: Vec<Scalar> = (0..num_global)
        .map(|i| {
            gradient_global[i]
                - (0..num_local)
                    .map(|j| coupling_global[i][j] * decrement_local[j])
                    .sum::<Scalar>()
        })
        .collect();
    let local_decrease = 0.5
        * gradient_local
            .iter()
            .zip(decrement_local.iter())
            .map(|(gradient, decrement)| gradient * decrement)
            .sum::<Scalar>();
    let mut accepted = false;
    for _ in 0..TRUST_REGION_MAX_REJECTIONS {
        *decrement_outer = more_sorensen_dense(tangent_outer, update_outer, *radius, num_global);
        decrement_inner
            .iter_mut()
            .zip(decrement_local.iter())
            .for_each(|(decrement_inner_i, decrement_local_i)| {
                *decrement_inner_i = *decrement_local_i
            });
        (0..num_global).for_each(|k| {
            decrement_inner
                .iter_mut()
                .zip(eliminated[k].iter())
                .for_each(|(decrement_inner_i, eliminated_ki)| {
                    *decrement_inner_i -= eliminated_ki * decrement_outer[k]
                })
        });
        *penalty = raise_penalty(
            *penalty,
            multipliers_global
                .iter()
                .zip(decrement_outer.iter().skip(num_global))
                .chain(
                    multipliers_local
                        .iter()
                        .zip(decrement_inner.iter().skip(num_local)),
                ),
        );
        let violated = *penalty
            * (violation(constraint_matrix_global, constraint_rhs_global, &*global)
                + violation(constraint_matrix_local, constraint_rhs_local, &*local));
        let value = function(&*global, &*local)? + violated;
        let predicted = merit_slope(
            reduced_gradient.iter().zip(decrement_outer.iter()),
            violated,
        ) - 0.5 * tangent_outer.quadratic_form_leading(decrement_outer, num_global)
            + local_decrease;
        if predicted <= Scalar::EPSILON * value.abs() {
            accepted = true;
            break;
        }
        let mut trial_global = global.clone();
        let mut trial_local = local.clone();
        let mut trial_multipliers_global = multipliers_global.clone();
        let mut trial_multipliers_local = multipliers_local.clone();
        trial_global.decrement_from_chained(&mut trial_multipliers_global, decrement_outer);
        trial_local.decrement_from_chained(&mut trial_multipliers_local, decrement_inner);
        let primal_norm = Norm::Euclidean.over(decrement_outer.iter().take(num_global).copied());
        //
        // A trial the problem cannot be evaluated at is a step
        // rejected, not an error: it is the very overreach the
        // radius exists to rein in.
        //
        let Ok(value_trial) = function(&trial_global, &trial_local).map(|value| {
            value
                + *penalty
                    * (violation(
                        constraint_matrix_global,
                        constraint_rhs_global,
                        &trial_global,
                    ) + violation(constraint_matrix_local, constraint_rhs_local, &trial_local))
        }) else {
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
    global.decrement_from_chained(multipliers_global, decrement_outer);
    local.decrement_from_chained(multipliers_local, decrement_inner);
    Ok(())
}
