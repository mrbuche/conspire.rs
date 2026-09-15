#[cfg(test)]
mod test;

use super::{Integrable, RECONSTRUCT_FAILED};
use crate::math::{
    Derivative, Differentiable, Quantity, TensorVec,
    integrate::{IntegrationError, Times},
};
use std::ops::Mul;

/// Explicit Euler for a single [`Integrable`], one step per interval of `time`.
///
/// ```math
/// \mathbf{x}_{n+1} = \mathrm{reconstruct}\!\left(\mathbf{x}_n,\ h\,\mathbf{f}(t_n, \mathbf{x}_n)\right)
/// ```
pub fn integrate_euler<Field, U, T>(
    mut rate: impl FnMut(Quantity<T>, &Field::Point) -> Result<Derivative<Field::Increment, T>, String>,
    time: &[Quantity<T>],
    initial_condition: Field::Point,
) -> Result<(Times<T>, U), IntegrationError>
where
    Field: Integrable,
    Field::Point: Clone,
    Field::Increment: Differentiable<T>,
    for<'a> &'a Derivative<Field::Increment, T>: Mul<Quantity<T>, Output = Field::Increment>,
    U: TensorVec<Item = Field::Point>,
{
    let mut point = initial_condition;
    let mut points = U::new();
    let mut times = Times::new();
    points.push(point.clone());
    times.push(time[0]);
    for step in time.windows(2) {
        let increment = &rate(step[0], &point)? * (step[1] - step[0]);
        point = Field::reconstruct(&point, &increment)
            .map_err(|_| IntegrationError::from(RECONSTRUCT_FAILED.to_string()))?;
        points.push(point.clone());
        times.push(step[1]);
    }
    Ok((times, points))
}
