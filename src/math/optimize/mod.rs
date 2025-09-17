#[cfg(test)]
mod test;

mod constraint;
mod gradient_descent;
mod line_search;
mod newton_raphson;

pub use constraint::EqualityConstraint;
pub use gradient_descent::GradientDescent;
pub use line_search::{LineSearch, LineSearchError};
pub use newton_raphson::NewtonRaphson;

use crate::{
    defeat_message,
    math::{
        Jacobian, Scalar, Solution, TestError,
        integrate::IntegrationError,
        matrix::square::{Banded, SquareMatrixError},
    },
};
use std::{
    fmt::{self, Debug, Display, Formatter},
    ops::Mul,
};

/// Zeroth-order root-finding algorithms.
pub trait ZerothOrderRootFinding<X> {
    fn root(
        &self,
        function: impl Fn(&X) -> Result<X, OptimizationError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizationError>;
}

/// First-order root-finding algorithms.
pub trait FirstOrderRootFinding<F, J, X> {
    fn root(
        &self,
        function: impl Fn(&X) -> Result<F, OptimizationError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizationError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizationError>;
}

/// First-order optimization algorithms.
pub trait FirstOrderOptimization<F, X> {
    fn minimize(
        &self,
        function: impl Fn(&X) -> Result<F, OptimizationError>,
        jacobian: impl Fn(&X) -> Result<X, OptimizationError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizationError>;
}

/// Second-order optimization algorithms.
pub trait SecondOrderOptimization<F, J, H, X> {
    fn minimize(
        &self,
        function: impl Fn(&X) -> Result<F, OptimizationError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizationError>,
        hessian: impl Fn(&X) -> Result<H, OptimizationError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
        banded: Option<Banded>,
    ) -> Result<X, OptimizationError>;
}

trait BacktrackingLineSearch<J, X>
where
    Self: Debug,
{
    fn backtracking_line_search(
        &self,
        function: impl Fn(&X) -> Result<Scalar, OptimizationError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizationError>,
        argument: &X,
        jacobian0: &J,
        decrement: &X,
        step_size: Scalar,
    ) -> Result<Scalar, OptimizationError>
    where
        J: Jacobian,
        for<'a> &'a J: From<&'a X>,
        X: Solution,
        for<'a> &'a X: Mul<Scalar, Output = X>,
    {
        if matches!(self.get_line_search(), LineSearch::None) {
            Ok(step_size)
        } else {
            match self.get_line_search().backtrack(
                &function, &jacobian, argument, jacobian0, decrement, step_size,
            ) {
                Ok(step_size) => Ok(step_size),
                Err(error) => Err(self.convert_error(error)),
            }
        }
    }
    fn convert_error(&self, error: LineSearchError) -> OptimizationError {
        OptimizationError::LineSearch(format!("{error}"), format!("{self:?}"))
    }
    fn get_line_search(&self) -> &LineSearch;
}

/// Possible errors encountered during optimization.
pub enum OptimizationError {
    Generic(String),
    LineSearch(String, String),
    MaximumStepsReached(usize, String),
    NotMinimum(String, String),
    SingularMatrix,
}

impl Debug for OptimizationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Generic(message) => message.to_string(),
            Self::LineSearch(error, solver) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In solver: {solver}."
                )
            }
            Self::MaximumStepsReached(steps, solver) => {
                format!(
                    "\x1b[1;91mMaximum number of steps ({steps}) reached.\x1b[0;91m\n\
                     In solver: {solver}."
                )
            }
            Self::NotMinimum(solution, solver) => {
                format!(
                    "\x1b[1;91mThe obtained solution is not a minimum.\x1b[0;91m\n\
                     For solution: {solution}.\n\
                     In solver: {solver}."
                )
            }
            Self::SingularMatrix => "\x1b[1;91mMatrix is singular.".to_string(),
        };
        write!(f, "\n{error}\n\x1b[0;2;31m{}\x1b[0m\n", defeat_message())
    }
}

impl Display for OptimizationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Generic(message) => message.to_string(),
            Self::LineSearch(error, solver) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In solver: {solver}."
                )
            }
            Self::MaximumStepsReached(steps, solver) => {
                format!(
                    "\x1b[1;91mMaximum number of steps ({steps}) reached.\x1b[0;91m\n\
                     In solver: {solver}."
                )
            }
            Self::NotMinimum(solution, solver) => {
                format!(
                    "\x1b[1;91mThe obtained solution is not a minimum.\x1b[0;91m\n\
                     For solution: {solution}.\n\
                     In solver: {solver}."
                )
            }
            Self::SingularMatrix => "\x1b[1;91mMatrix is singular.".to_string(),
        };
        write!(f, "{error}\x1b[0m")
    }
}

impl From<OptimizationError> for String {
    fn from(error: OptimizationError) -> Self {
        error.to_string()
    }
}

impl From<OptimizationError> for TestError {
    fn from(error: OptimizationError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl From<IntegrationError> for OptimizationError {
    fn from(_error: IntegrationError) -> Self {
        todo!()
    }
}

impl From<SquareMatrixError> for OptimizationError {
    fn from(_error: SquareMatrixError) -> Self {
        Self::SingularMatrix
    }
}
