#[cfg(test)]
mod test;

use super::{Integrable, reconstruct_or_err};
use crate::math::{Quantity, TensorVec, integrate::IntegrationError};
use crate::units::Time;

/// Cubic Hermite dense output over one accepted step, built in the field's Lie
/// algebra rather than on the state itself.
///
/// The flat interpolant combines `y_{n}`, `y_{n+1}` and the two end rates
/// affinely. That is meaningless for a group-valued state: the two states are
/// different group elements and the two rates live in different tangent spaces,
/// so the combination leaves the manifold (`det F_p` drifts) exactly the way the
/// additive march did.
///
/// Anchor everything at the left endpoint instead. The step already produced
/// `sigma` with `reconstruct(base, sigma) = y_{n+1}`, and the algebra is a flat
/// vector space — so the Hermite polynomial is built *there*,
///
/// ```math
/// \sigma(\theta) = h_{10}(\theta)\,\dot\sigma_0 + h_{01}(\theta)\,\sigma
///                + h_{11}(\theta)\,\dot\sigma_1
/// ,\qquad
/// \mathbf{y}(\theta) = \mathrm{reconstruct}(\mathbf{y}_n, \sigma(\theta))
/// ```
///
/// with `σ(0) = 0` and `σ(1) = sigma`, so both endpoints are reproduced exactly
/// and every interior point is an `expm` of a trace-free element — on the group
/// by construction. `slope_0` is the raw stage-0 slope (`dexpinv` at zero
/// displacement is the identity); `slope_1` is the endpoint rate pulled back
/// through [`Integrable::dexpinv`] at `sigma`, both already scaled by the
/// step. On a [`super::Flat`] field `reconstruct` adds and `dexpinv` is the
/// identity, and `h_{00} + h_{01} = 1` collapses this to the usual flat formula.
pub struct HermiteSegment<Field: Integrable, T = Time> {
    t_0: Quantity<T>,
    h: Quantity<T>,
    base: Field::Point,
    sigma: Field::Increment,
    slope_0: Field::Increment,
    slope_1: Field::Increment,
}

impl<Field, T> HermiteSegment<Field, T>
where
    Field: Integrable,
{
    /// A segment of the accepted step `[t_0, t_0 + h]` from `base`, the algebra
    /// displacement `sigma` over it, and the step-scaled algebra rates at its
    /// two ends (`slope_1` already pulled back through
    /// [`Integrable::dexpinv`] at `sigma`).
    pub fn new(
        t_0: Quantity<T>,
        h: Quantity<T>,
        base: Field::Point,
        sigma: Field::Increment,
        slope_0: Field::Increment,
        slope_1: Field::Increment,
    ) -> Self {
        Self {
            t_0,
            h,
            base,
            sigma,
            slope_0,
            slope_1,
        }
    }
    /// The state at `time`, reconstructed from the algebra Hermite polynomial.
    pub fn evaluate(&self, time: Quantity<T>) -> Result<Field::Point, IntegrationError> {
        let theta = (time - self.t_0).value() / self.h.value();
        let theta_2 = theta * theta;
        let theta_3 = theta_2 * theta;
        let mut increment = self.slope_0.clone() * (theta_3 - 2.0 * theta_2 + theta);
        increment += self.sigma.clone() * (3.0 * theta_2 - 2.0 * theta_3);
        increment += self.slope_1.clone() * (theta_3 - theta_2);
        reconstruct_or_err::<Field>(&self.base, &increment)
    }
}

/// Evaluates `segments` at `time_k`, in the segment that contains it (the last
/// one for a time past the final accepted step).
pub(super) fn hermite_at<Field, T>(
    segments: &[HermiteSegment<Field, T>],
    time_k: Quantity<T>,
) -> Result<Field::Point, IntegrationError>
where
    Field: Integrable,
{
    segments
        .iter()
        .find(|segment| time_k <= segment.t_0 + segment.h)
        .unwrap_or(&segments[segments.len() - 1])
        .evaluate(time_k)
}

/// Finds the containing segment and evaluates it there, over a whole grid of
/// requested times.
pub fn interpolate_hermite<Field, U, T>(
    segments: &[HermiteSegment<Field, T>],
    time: &[Quantity<T>],
) -> Result<U, IntegrationError>
where
    Field: Integrable,
    U: TensorVec<Item = Field::Point>,
{
    let mut points = U::new();
    for time_k in time {
        points.push(hermite_at::<Field, T>(segments, *time_k)?);
    }
    Ok(points)
}
