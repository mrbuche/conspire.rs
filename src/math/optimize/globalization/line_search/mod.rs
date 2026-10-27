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

use crate::{
    math::{Quantity, Scalar, Style, StyledError, styled_error},
    units::{Dimensionless, UnitMul},
};
use std::fmt::{Debug, Display};

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
    /// Backtrack along a step of the objective, whose slope along it is known.
    fn search<W, U, V>(
        &self,
        value: impl FnMut(Quantity<W>) -> Result<Quantity<V>, String>,
        slope: impl FnMut(Quantity<W>) -> Result<Quantity<U>, String>,
        step_size: Quantity<W>,
    ) -> Result<Quantity<W>, LineSearchError<W>>
    where
        W: UnitMul<U, Output = V>;
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

fn exhausted<W>(search: &impl Debug, max_steps: usize) -> LineSearchError<W> {
    LineSearchError::MaximumStepsReached(format!("{search:?}"), max_steps)
}

/// The value and the slope at no step, once the step is vetted.
fn start<W, U, V>(
    search: &impl Debug,
    value: &mut impl FnMut(Quantity<W>) -> Result<Quantity<V>, String>,
    slope: &mut impl FnMut(Quantity<W>) -> Result<Quantity<U>, String>,
    step_size: Quantity<W>,
) -> Result<(Quantity<V>, Quantity<U>), LineSearchError<W>> {
    if step_size <= Quantity::new(0.0) {
        return Err(LineSearchError::NegativeStepSize(
            format!("{search:?}"),
            step_size,
        ));
    }
    let (Ok(value), Ok(slope)) = (value(Quantity::new(0.0)), slope(Quantity::new(0.0))) else {
        return Err(LineSearchError::InvalidStartingPoint(format!("{search:?}")));
    };
    if slope <= Quantity::new(0.0) {
        return Err(LineSearchError::NotDescentDirection(format!("{search:?}")));
    }
    Ok((value, slope))
}

/// Cut the step back until the decrease it buys is accepted.
///
/// The decrease is what the objective loses, so a step that cannot be
/// evaluated is cut back as one that buys nothing. It is judged against what
/// the slope along the step promises.
fn backtrack<W, U, V>(
    search: &impl Debug,
    cut_back: Scalar,
    max_steps: usize,
    mut value: impl FnMut(Quantity<W>) -> Result<Quantity<V>, String>,
    mut slope: impl FnMut(Quantity<W>) -> Result<Quantity<U>, String>,
    step_size: Quantity<W>,
    accepted: impl Fn(Quantity<V>, Quantity<V>) -> bool,
) -> Result<Quantity<W>, LineSearchError<W>>
where
    W: UnitMul<U, Output = V>,
{
    let (initial, gradient) = start(search, &mut value, &mut slope, step_size)?;
    let mut n = step_size;
    for _ in 0..max_steps {
        if let Ok(trial) = value(n)
            && accepted(initial - trial, n * gradient)
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
    accepted: impl Fn(Scalar, Scalar) -> bool,
) -> Result<Scalar, LineSearchError> {
    if step_size <= 0.0 {
        return Err(LineSearchError::NegativeStepSize(
            format!("{search:?}"),
            Quantity::new(step_size),
        ));
    }
    if slope <= 0.0 {
        return Err(LineSearchError::NotDescentDirection(format!("{search:?}")));
    }
    let mut n = step_size;
    for _ in 0..max_steps {
        if let Ok(trial) = merit(n)
            && accepted(value - trial, n * slope)
        {
            return Ok(n);
        } else {
            n *= cut_back
        }
    }
    Err(exhausted(search, max_steps))
}

/// Possible errors encountered during line search.
pub enum LineSearchError<W = Dimensionless> {
    InvalidStartingPoint(String),
    MaximumStepsReached(String, usize),
    NegativeStepSize(String, Quantity<W>),
    NotDescentDirection(String),
}

impl<W> StyledError for LineSearchError<W> {
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

styled_error!(<W> LineSearchError<W>);
