use crate::math::{
    Scalar, Vector,
    optimize::{NewtonRaphson, OptimizationError, newton_raphson::backtrack::Backtrack},
};
use std::{fmt::Display, ops::Mul};

pub(super) const PENALTY_SAFETY: Scalar = 2.0;

pub(super) fn violation<M, T>(
    constraint_matrix: &M,
    constraint_rhs: &Vector,
    variables: &T,
) -> Scalar
where
    for<'a> &'a M: Mul<&'a T, Output = Vector>,
{
    (constraint_rhs - constraint_matrix * variables)
        .into_iter()
        .map(|entry| entry.abs())
        .sum()
}

/// Raises the penalty until it outweighs every multiplier the step would reach.
///
/// The penalty only ever climbs, so a step taken where the multipliers were
/// larger is not undone by one taken where they are smaller.
pub(super) fn raise_penalty<'a>(
    penalty: Scalar,
    multipliers: impl Iterator<Item = (&'a Scalar, &'a Scalar)>,
) -> Scalar {
    penalty.max(
        PENALTY_SAFETY
            * multipliers.fold(0.0, |largest: Scalar, (multiplier, decrement)| {
                largest.max((multiplier - decrement).abs())
            }),
    )
}

/// The slope of the merit function along the step.
///
/// Stated in the sign the line search reads, where a positive slope is the
/// decrease a step buys, so the violation the penalty charges for adds to the
/// descent the gradient promises rather than subtracting from it.
pub(super) fn merit_slope<'a>(
    gradients: impl Iterator<Item = (&'a Scalar, &'a Scalar)>,
    violated: Scalar,
) -> Scalar {
    gradients
        .map(|(gradient, decrement)| gradient * decrement)
        .sum::<Scalar>()
        + violated
}

/// Backtracks on the merit function, or takes the whole step where there is
/// nothing to backtrack along.
///
/// The line search refuses a direction that is not one of descent, and cannot
/// resolve one whose descent is finer than the merit it would be measured
/// against: the conditions all compare a decrease that the subtraction has
/// already rounded away. Either is met by stepping whole rather than by
/// asking and being turned away.
///
/// A slope is a merit against a step, so it is the merit it is judged
/// against, which leaves the comparison a ratio and the threshold the
/// precision that ratio is held in. A number of its own would carry units and
/// mean something different at every scale.
pub(super) fn backtrack_penalty<L, S>(
    newton_raphson: &NewtonRaphson<L, S>,
    search: &impl Backtrack,
    merit: impl FnMut(Scalar) -> Result<Scalar, String>,
    value: Scalar,
    slope: Scalar,
) -> Result<Scalar, OptimizationError>
where
    S: Display,
{
    if slope <= Scalar::EPSILON * value.abs() {
        Ok(1.0)
    } else {
        search
            .merit(merit, value, slope, 1.0)
            .map_err(|error| OptimizationError::upstream(error, newton_raphson))
    }
}
