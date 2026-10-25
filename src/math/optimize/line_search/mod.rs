#[cfg(test)]
mod test;

mod armijo;
mod error;
mod goldstein;
mod none;
mod residual;
mod wolfe;

pub use armijo::Armijo;
pub use error::Error;
pub use goldstein::Goldstein;
pub use none::NoSearch;
pub use residual::Residual;
pub use wolfe::Wolfe;

use crate::math::{
    Erase, Jacobian, Quantity, Scalar, Solution, Style, StyledError, Tensor, styled_error,
};
use std::{
    fmt::{Debug, Display},
    ops::Mul,
};

/// Line searches that judge a step by a merit built from the objective.
///
/// The objective is the function being minimized, plus a penalty where it is
/// constrained, so these are the searches a minimization can take and a root
/// cannot: a root has no objective to compare the values of.
pub trait ObjectiveMerit: Clone + Debug + Display {
    /// Whether anything is searched at all.
    const SEARCHES: bool = true;
    /// Whether the value and slope of the merit are read, and so worth preparing.
    const MERIT: bool = true;
    /// Backtrack along a step of the objective, whose gradient is known.
    fn search<X, J, D, W, E>(
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
    /// Backtrack on a merit function of the step size alone.
    ///
    /// The exact penalty function is not differentiable, its norm having a kink
    /// wherever a constraint is satisfied, so its slope along the step is
    /// supplied rather than recovered from a gradient.
    fn search_merit(
        &self,
        merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        value: Scalar,
        slope: Scalar,
        step_size: Scalar,
    ) -> Result<Scalar, LineSearchError>;
}

/// Line searches that judge a step by a merit built from the residual.
///
/// The merit is half the squared norm of the residual being driven to zero.
/// Along an exact Newton step its slope is the negative of twice itself, so
/// nothing beyond the residual need be known.
pub trait ResidualMerit: Clone + Debug + Display {
    /// Whether anything is searched at all.
    const SEARCHES: bool = true;
    /// Whether the value of the merit is read, and so worth preparing.
    const MERIT: bool = true;
    /// Backtrack on the residual merit of the step size alone.
    ///
    /// The step is judged along the share `fraction` of a step, which is
    /// one unless something shortened the step beforehand.
    fn search_residual(
        &self,
        merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        value: Scalar,
        fraction: Scalar,
    ) -> Result<Scalar, LineSearchError>;
}

fn exhausted(search: &impl Debug, max_steps: usize) -> LineSearchError {
    LineSearchError::MaximumStepsReached(format!("{search:?}"), max_steps)
}

/// The value and the slope at no step, once the step and direction are vetted.
fn start<X, J, D, E>(
    search: &impl Debug,
    function: &mut impl FnMut(&X, Scalar) -> Result<Scalar, String>,
    argument: &X,
    jacobian0: &J,
    decrement: &D,
    step_size: Scalar,
) -> Result<(Scalar, Scalar), LineSearchError>
where
    J: Erase<Erased = E> + Jacobian,
    D: Erase<Erased = E>,
    E: Tensor,
{
    if step_size <= 0.0 {
        return Err(LineSearchError::NegativeStepSize(
            format!("{search:?}"),
            step_size,
        ));
    }
    let Ok(value) = function(argument, 0.0) else {
        return Err(LineSearchError::InvalidStartingPoint(format!("{search:?}")));
    };
    let slope = jacobian0.erase().full_contraction(decrement.erase());
    if slope <= 0.0 {
        return Err(LineSearchError::NotDescentDirection(format!("{search:?}")));
    }
    Ok((value, slope))
}

/// Cut the step back until the decrease it buys is accepted.
///
/// The decrease is what the objective loses, so a step that cannot be
/// evaluated is cut back as one that buys nothing.
#[expect(clippy::too_many_arguments)]
fn backtrack<X, J, D, W, E>(
    search: &impl Debug,
    cut_back: Scalar,
    max_steps: usize,
    mut function: impl FnMut(&X, Scalar) -> Result<Scalar, String>,
    argument: &X,
    jacobian0: &J,
    decrement: &D,
    step_size: Scalar,
    accepted: impl Fn(Scalar, Scalar, Scalar) -> bool,
) -> Result<Scalar, LineSearchError>
where
    J: Erase<Erased = E> + Jacobian,
    D: Erase<Erased = E>,
    E: Tensor,
    X: Solution,
    for<'a> &'a D: Mul<Quantity<W>, Output = X>,
{
    let (value, slope) = start(
        search,
        &mut function,
        argument,
        jacobian0,
        decrement,
        step_size,
    )?;
    let mut n = step_size;
    for _ in 0..max_steps {
        if let Ok(trial) = function(&(decrement * Quantity::new(-n) + argument), n)
            && accepted(value - trial, n, slope)
        {
            return Ok(n);
        } else {
            n *= cut_back
        }
    }
    Err(exhausted(search, max_steps))
}

/// Cut the step back until the decrease of the merit is accepted.
#[expect(clippy::too_many_arguments)]
fn backtrack_merit(
    search: &impl Debug,
    cut_back: Scalar,
    max_steps: usize,
    mut merit: impl FnMut(Scalar) -> Result<Scalar, String>,
    value: Scalar,
    slope: Scalar,
    step_size: Scalar,
    accepted: impl Fn(Scalar, Scalar, Scalar) -> bool,
) -> Result<Scalar, LineSearchError> {
    if step_size <= 0.0 {
        return Err(LineSearchError::NegativeStepSize(
            format!("{search:?}"),
            step_size,
        ));
    }
    if slope <= 0.0 {
        return Err(LineSearchError::NotDescentDirection(format!("{search:?}")));
    }
    let mut n = step_size;
    for _ in 0..max_steps {
        if let Ok(trial) = merit(n)
            && accepted(value - trial, n, slope)
        {
            return Ok(n);
        } else {
            n *= cut_back
        }
    }
    Err(exhausted(search, max_steps))
}

/// Possible errors encountered during line search.
pub enum LineSearchError {
    InvalidStartingPoint(String),
    MaximumStepsReached(String, usize),
    NegativeStepSize(String, Scalar),
    NotDescentDirection(String),
}

impl StyledError for LineSearchError {
    fn message(&self, style: &Style) -> String {
        let (h, c) = (style.headline, style.frame);
        match self {
            Self::InvalidStartingPoint(line_search) => format!(
                "{h}Starting point is invalid.{c}\n\
                In line search: {line_search}."
            ),
            Self::MaximumStepsReached(line_search, steps) => format!(
                "{h}Maximum number of steps ({steps}) reached.{c}\n\
                In line search: {line_search}."
            ),
            Self::NegativeStepSize(line_search, step_size) => format!(
                "{h}Negative step size ({step_size}) encountered.{c}\n\
                In line search: {line_search}."
            ),
            Self::NotDescentDirection(line_search) => format!(
                "{h}Direction is not a descent direction.{c}\n\
                In line search: {line_search}."
            ),
        }
    }
}

styled_error!(LineSearchError);
