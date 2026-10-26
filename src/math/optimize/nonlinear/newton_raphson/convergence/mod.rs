use crate::{
    math::{Quantity, Scalar, Tensor, Vector, optimize::TrustRegion},
    units::UnitRatio,
};

use super::NewtonRaphson;

/// Whether every block of the residual has come within the tolerance.
///
/// Each block is measured on its own against a tolerance of its own, the
/// constraint violation being of another kind entirely from the residual, and
/// either kind in one block from that in another.
///
/// The scales are what each block was on the first step, so that the relative
/// tolerance is compared against a ratio of two norms of the same kind, and
/// means the same thing whatever units that kind is measured in.
pub(crate) fn converged<L, S>(
    newton_raphson: &NewtonRaphson<L, S>,
    residual: &Vector,
    variables: usize,
    scales: &mut Option<(Scalar, Scalar)>,
) -> bool {
    let norms = (
        newton_raphson
            .error_norm
            .over(residual.iter().take(variables).copied()),
        newton_raphson
            .error_norm
            .over(residual.iter().skip(variables).copied()),
    );
    let scales = scales.get_or_insert(norms);
    let met = |norm: Scalar, scale: Scalar, abs_tol: Scalar| {
        norm < abs_tol
            || newton_raphson
                .rel_tol
                .is_some_and(|rel_tol| norm / scale < rel_tol)
    };
    met(norms.0, scales.0, newton_raphson.abs_tol.residual)
        && met(norms.1, scales.1, newton_raphson.abs_tol.constraint)
}

/// Shortens a decrement until it moves the variables no further than the maximum.
///
/// The maximum is read back in the unit of the decrement. What is returned is
/// the share of the decrement that was kept.
pub(crate) fn shorten<L, S, X>(newton_raphson: &NewtonRaphson<L, S>, decrement: &mut X) -> Scalar
where
    X: Tensor,
    <X as Tensor>::Unit: UnitRatio,
{
    if let TrustRegion::Fixed { radius, norm } = newton_raphson.trust_region {
        let size = norm.apply(decrement);
        let radius = Quantity::new(radius);
        if size > radius {
            let share = Scalar::from(radius / size);
            *decrement *= share;
            return share;
        }
    }
    1.0
}

/// Shortens the step until the variables move no further than the maximum.
///
/// Only the variables are measured, the multipliers being of another kind
/// entirely, but everything is scaled together so that the direction survives.
///
/// What is returned is the share of the decrement that was kept.
pub(crate) fn limit_decrement<L, S>(
    newton_raphson: &NewtonRaphson<L, S>,
    decrements: &mut [(&mut Vector, usize)],
) -> Scalar {
    if let TrustRegion::Fixed { radius, norm } = newton_raphson.trust_region {
        let size = norm.over(
            decrements
                .iter()
                .flat_map(|(decrement, variables)| decrement.iter().take(*variables).copied()),
        );
        if size > radius {
            decrements
                .iter_mut()
                .for_each(|(decrement, _)| **decrement *= radius / size);
            return radius / size;
        }
    }
    1.0
}
