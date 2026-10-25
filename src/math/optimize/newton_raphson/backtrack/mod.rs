use super::super::{
    super::{Erase, Jacobian, Scalar, Solution, Tensor},
    LineSearchError, ObjectiveMerit, OptimizationError, ResidualMerit,
};
use crate::math::{Norm, Quantity};
use std::{fmt::Display, ops::Mul};

use super::NewtonRaphson;

/// A line search as the Newton loops take it, in the problem it is taken in.
///
/// A loop is shared by minimization and root finding, which judge a step by
/// different merits and so by different kinds of line search. What it holds
/// is the one kind its problem allows, so asking a root for an objective, or
/// a minimization for a residual, is not a question a loop can put.
pub(super) trait Backtrack {
    /// Whether the merit is built from the objective, rather than the residual.
    const OBJECTIVE: bool;
    /// Whether anything is searched at all.
    const SEARCHES: bool;
    /// Whether the value and slope of the merit are read.
    const MERIT: bool;
    /// Backtrack on a merit of the step size alone.
    ///
    /// A residual merit has its slope along the share `fraction` of a Newton
    /// step, so is not given one.
    fn merit(
        &self,
        merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        value: Scalar,
        slope: Scalar,
        fraction: Scalar,
    ) -> Result<Scalar, LineSearchError>;
    /// Backtrack along a step of the objective, whose gradient is known.
    fn gradient<X, J, D, W, E>(
        &self,
        function: impl FnMut(&X, Scalar) -> Result<Scalar, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        argument: &X,
        jacobian0: &J,
        decrement: &D,
        step_size: Scalar,
    ) -> Result<Scalar, LineSearchError>
    where
        J: Erase<Erased = E> + Jacobian,
        D: Erase<Erased = E>,
        E: Tensor,
        X: Solution,
        for<'a> &'a D: Mul<Quantity<W>, Output = X>;
}

/// The line search of a minimization.
pub(super) struct Objective<'a, S>(pub(super) &'a S);

/// The line search of a root.
pub(super) struct Residually<'a, S>(pub(super) &'a S);

impl<S> Backtrack for Objective<'_, S>
where
    S: ObjectiveMerit,
{
    const OBJECTIVE: bool = true;
    const SEARCHES: bool = S::SEARCHES;
    const MERIT: bool = S::MERIT;
    fn merit(
        &self,
        merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        value: Scalar,
        slope: Scalar,
        _fraction: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        self.0.search_merit(merit, value, slope, 1.0)
    }
    fn gradient<X, J, D, W, E>(
        &self,
        function: impl FnMut(&X, Scalar) -> Result<Scalar, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        argument: &X,
        jacobian0: &J,
        decrement: &D,
        step_size: Scalar,
    ) -> Result<Scalar, LineSearchError>
    where
        J: Erase<Erased = E> + Jacobian,
        D: Erase<Erased = E>,
        E: Tensor,
        X: Solution,
        for<'a> &'a D: Mul<Quantity<W>, Output = X>,
    {
        self.0.search(
            function, jacobian, argument, jacobian0, decrement, step_size,
        )
    }
}

impl<S> Backtrack for Residually<'_, S>
where
    S: ResidualMerit,
{
    const OBJECTIVE: bool = false;
    const SEARCHES: bool = S::SEARCHES;
    const MERIT: bool = S::MERIT;
    fn merit(
        &self,
        merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        value: Scalar,
        _slope: Scalar,
        fraction: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        self.0.search_residual(merit, value, fraction)
    }
    fn gradient<X, J, D, W, E>(
        &self,
        _function: impl FnMut(&X, Scalar) -> Result<Scalar, String>,
        _jacobian: impl FnMut(&X) -> Result<J, String>,
        _argument: &X,
        _jacobian0: &J,
        _decrement: &D,
        _step_size: Scalar,
    ) -> Result<Scalar, LineSearchError>
    where
        J: Erase<Erased = E> + Jacobian,
        D: Erase<Erased = E>,
        E: Tensor,
        X: Solution,
        for<'a> &'a D: Mul<Quantity<W>, Output = X>,
    {
        unreachable!("A root has no objective to take the gradient of.")
    }
}

pub(super) fn half_square_norm(tensor: &impl Tensor) -> Scalar {
    0.5 * Norm::Euclidean.measure(tensor).powi(2)
}

pub(super) fn half_square(entries: impl Iterator<Item = Scalar>) -> Scalar {
    0.5 * entries.map(|entry| entry * entry).sum::<Scalar>()
}

pub(super) fn searched<L, S>(
    newton_raphson: &NewtonRaphson<L, S>,
    step_size: Result<Scalar, LineSearchError>,
) -> Result<Scalar, OptimizationError>
where
    S: Display,
{
    step_size.map_err(|error| OptimizationError::upstream(error, newton_raphson))
}

/// The step size along the share `fraction` of a Newton step, judged by the
/// residual that trial points leave.
///
/// What a trial leaves is half the squared norm of the residual there, or only
/// whether there is one where the search reads no merit. The value at no step
/// is only asked for by a search that reads it.
pub(super) fn residual_step<B, L, S>(
    newton_raphson: &NewtonRaphson<L, S>,
    search: &B,
    value: impl FnOnce() -> Scalar,
    leaves: impl FnMut(Scalar) -> Result<Scalar, String>,
    fraction: Scalar,
) -> Result<Scalar, OptimizationError>
where
    B: Backtrack,
    S: Display,
{
    let value = if B::MERIT { value() } else { 0.0 };
    searched(newton_raphson, search.merit(leaves, value, 0.0, fraction))
}
