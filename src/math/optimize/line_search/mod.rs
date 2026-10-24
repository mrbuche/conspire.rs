#[cfg(test)]
mod test;

use crate::math::{
    Erase, Jacobian, Quantity, Scalar, Solution, Style, StyledError, Tensor, styled_error,
};
use std::{
    fmt::{self, Debug, Display, Formatter},
    ops::Mul,
};

/// Line searches that judge a step by a merit built from the objective.
///
/// The objective is the function being minimized, plus a penalty where it is
/// constrained, so these are the searches a minimization can take and a root
/// cannot: a root has no objective to compare the values of.
///
/// A minimization takes one of them:
///
/// ```
/// use conspire::math::optimize::{Armijo, EqualityConstraint, NewtonRaphson, Optimization};
/// NewtonRaphson::default()
///     .with_line_search(Armijo::default())
///     .minimize(
///         |x: &f64| Ok(x.powi(2) / 2.0),
///         |x: &f64| Ok(*x),
///         |_: &f64| Ok(1.0),
///         1.0,
///         vec![EqualityConstraint::None],
///         None,
///     )
///     .unwrap();
/// ```
///
/// A root cannot:
///
/// ```compile_fail
/// use conspire::math::optimize::{Armijo, EqualityConstraint, NewtonRaphson, RootFinding};
/// NewtonRaphson::default()
///     .with_line_search(Armijo::default())
///     .root(
///         |x: &f64| Ok(*x),
///         |_: &f64| Ok(1.0),
///         1.0,
///         vec![EqualityConstraint::None],
///         None,
///     )
///     .unwrap();
/// ```
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
///
/// A root takes one of them:
///
/// ```
/// use conspire::math::optimize::{EqualityConstraint, NewtonRaphson, Residual, RootFinding};
/// NewtonRaphson::default()
///     .with_line_search(Residual::default())
///     .root(
///         |x: &f64| Ok(*x),
///         |_: &f64| Ok(1.0),
///         1.0,
///         vec![EqualityConstraint::None],
///         None,
///     )
///     .unwrap();
/// ```
///
/// A minimization cannot:
///
/// ```compile_fail
/// use conspire::math::optimize::{EqualityConstraint, NewtonRaphson, Optimization, Residual};
/// NewtonRaphson::default()
///     .with_line_search(Residual::default())
///     .minimize(
///         |x: &f64| Ok(x.powi(2) / 2.0),
///         |x: &f64| Ok(*x),
///         |_: &f64| Ok(1.0),
///         1.0,
///         vec![EqualityConstraint::None],
///         None,
///     )
///     .unwrap();
/// ```
pub trait ResidualMerit: Clone + Debug + Display {
    /// Whether anything is searched at all.
    const SEARCHES: bool = true;
    /// Whether the value of the merit is read, and so worth preparing.
    const MERIT: bool = true;
    /// Backtrack on the residual merit of the step size alone.
    ///
    /// The step is judged along the share `fraction` of a Newton step, which is
    /// one unless something shortened the step beforehand.
    fn search_residual(
        &self,
        merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        value: Scalar,
        fraction: Scalar,
    ) -> Result<Scalar, LineSearchError>;
}

/// The Armijo condition on the objective.
#[derive(Clone, Debug)]
pub struct Armijo {
    pub control: Scalar,
    pub cut_back: Scalar,
    pub max_steps: usize,
}

impl Default for Armijo {
    fn default() -> Self {
        Self {
            control: 1e-3,
            cut_back: 9e-1,
            max_steps: 100,
        }
    }
}

/// Backtrack for errors.
#[derive(Clone, Debug)]
pub struct Error {
    pub cut_back: Scalar,
    pub max_steps: usize,
}

/// The Goldstein conditions on the objective.
#[derive(Clone, Debug)]
pub struct Goldstein {
    pub control: Scalar,
    pub cut_back: Scalar,
    pub max_steps: usize,
}

/// No line search.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoSearch;

/// The Armijo condition on half the squared norm of the residual.
#[derive(Clone, Debug)]
pub struct Residual {
    pub control: Scalar,
    pub cut_back: Scalar,
    pub max_steps: usize,
}

impl Default for Residual {
    fn default() -> Self {
        Self {
            control: 1e-3,
            cut_back: 9e-1,
            max_steps: 100,
        }
    }
}

/// The Wolfe conditions on the objective.
#[derive(Clone, Debug)]
pub struct Wolfe {
    pub control_1: Scalar,
    pub control_2: Scalar,
    pub cut_back: Scalar,
    pub max_steps: usize,
    pub strong: bool,
}

macro_rules! display {
    ($($search:ident),*) => {
        $(impl Display for $search {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                write!(f, stringify!($search))
            }
        })*
    };
}

display!(Armijo, Error, Goldstein, NoSearch, Residual, Wolfe);

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

impl ObjectiveMerit for Armijo {
    fn search<X, J, D, W, E>(
        &self,
        function: impl FnMut(&X, Scalar) -> Result<Scalar, String>,
        _jacobian: impl FnMut(&X) -> Result<J, String>,
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
        backtrack(
            self,
            self.cut_back,
            self.max_steps,
            function,
            argument,
            jacobian0,
            decrement,
            step_size,
            |decrease, n, slope| decrease >= n * self.control * slope,
        )
    }
    fn search_merit(
        &self,
        merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        value: Scalar,
        slope: Scalar,
        step_size: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        backtrack_merit(
            self,
            self.cut_back,
            self.max_steps,
            merit,
            value,
            slope,
            step_size,
            |decrease, n, slope| decrease >= n * self.control * slope,
        )
    }
}

impl ObjectiveMerit for Goldstein {
    fn search<X, J, D, W, E>(
        &self,
        function: impl FnMut(&X, Scalar) -> Result<Scalar, String>,
        _jacobian: impl FnMut(&X) -> Result<J, String>,
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
        backtrack(
            self,
            self.cut_back,
            self.max_steps,
            function,
            argument,
            jacobian0,
            decrement,
            step_size,
            |decrease, n, slope| self.accepts(decrease, n, slope),
        )
    }
    fn search_merit(
        &self,
        merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        value: Scalar,
        slope: Scalar,
        step_size: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        backtrack_merit(
            self,
            self.cut_back,
            self.max_steps,
            merit,
            value,
            slope,
            step_size,
            |decrease, n, slope| self.accepts(decrease, n, slope),
        )
    }
}

impl Goldstein {
    fn accepts(&self, decrease: Scalar, n: Scalar, slope: Scalar) -> bool {
        !(n * (1.0 - self.control) * slope < decrease || decrease < n * self.control * slope)
    }
}

impl ObjectiveMerit for Error {
    const MERIT: bool = false;
    fn search<X, J, D, W, E>(
        &self,
        function: impl FnMut(&X, Scalar) -> Result<Scalar, String>,
        _jacobian: impl FnMut(&X) -> Result<J, String>,
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
        backtrack(
            self,
            self.cut_back,
            self.max_steps,
            function,
            argument,
            jacobian0,
            decrement,
            step_size,
            |_, _, _| true,
        )
    }
    fn search_merit(
        &self,
        merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        _value: Scalar,
        _slope: Scalar,
        step_size: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        self.reachable(merit, step_size)
    }
}

impl Error {
    fn reachable(
        &self,
        mut merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        step_size: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        let mut n = step_size;
        for _ in 0..self.max_steps {
            if merit(n).is_ok() {
                return Ok(n);
            } else {
                n *= self.cut_back
            }
        }
        Err(exhausted(self, self.max_steps))
    }
}

impl ResidualMerit for Error {
    const MERIT: bool = false;
    fn search_residual(
        &self,
        merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        _value: Scalar,
        _fraction: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        self.reachable(merit, 1.0)
    }
}

impl ObjectiveMerit for NoSearch {
    const SEARCHES: bool = false;
    const MERIT: bool = false;
    fn search<X, J, D, W, E>(
        &self,
        _function: impl FnMut(&X, Scalar) -> Result<Scalar, String>,
        _jacobian: impl FnMut(&X) -> Result<J, String>,
        _argument: &X,
        _jacobian0: &J,
        _decrement: &D,
        step_size: Scalar,
    ) -> Result<Scalar, LineSearchError>
    where
        J: Erase<Erased = E> + Jacobian,
        D: Erase<Erased = E>,
        E: Tensor,
        X: Solution,
        for<'a> &'a D: Mul<Quantity<W>, Output = X>,
    {
        Ok(step_size)
    }
    fn search_merit(
        &self,
        _merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        _value: Scalar,
        _slope: Scalar,
        step_size: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        Ok(step_size)
    }
}

impl ResidualMerit for NoSearch {
    const SEARCHES: bool = false;
    const MERIT: bool = false;
    fn search_residual(
        &self,
        _merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        _value: Scalar,
        _fraction: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        Ok(1.0)
    }
}

impl ResidualMerit for Residual {
    fn search_residual(
        &self,
        mut merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        value: Scalar,
        fraction: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        let t = self.control * 2.0 * fraction * value;
        let mut n = 1.0;
        for _ in 0..self.max_steps {
            if let Ok(trial) = merit(n)
                && value - trial >= n * t
            {
                return Ok(n);
            } else {
                n *= self.cut_back
            }
        }
        Err(exhausted(self, self.max_steps))
    }
}

impl ObjectiveMerit for Wolfe {
    fn search<X, J, D, W, E>(
        &self,
        mut function: impl FnMut(&X, Scalar) -> Result<Scalar, String>,
        mut jacobian: impl FnMut(&X) -> Result<J, String>,
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
        let (value, slope) = start(
            self,
            &mut function,
            argument,
            jacobian0,
            decrement,
            step_size,
        )?;
        let t_1 = self.control_1 * slope;
        let t_2 = self.control_2 * slope;
        let mut n = step_size;
        for _ in 0..self.max_steps {
            let trial = decrement * Quantity::new(-n) + argument;
            if let Ok(trial_value) = function(&trial, n)
                && let Ok(trial_jacobian) = jacobian(&trial)
                && value - trial_value >= n * t_1
                && {
                    let curvature = trial_jacobian.erase().full_contraction(decrement.erase());
                    if self.strong {
                        curvature.abs() < t_2.abs()
                    } else {
                        curvature < t_2
                    }
                }
            {
                return Ok(n);
            } else {
                n *= self.cut_back
            }
        }
        Err(exhausted(self, self.max_steps))
    }
    fn search_merit(
        &self,
        _merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        _value: Scalar,
        _slope: Scalar,
        _step_size: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        panic!(
            "The Wolfe conditions need the gradient of the merit function, which the exact penalty function does not have."
        )
    }
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
