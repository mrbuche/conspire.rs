#[cfg(test)]
mod test;

mod constraint;
mod gradient_descent;
mod line_search;
mod newton_raphson;

pub use constraint::EqualityConstraint;
pub use gradient_descent::GradientDescent;
pub use line_search::LineSearch;
pub use newton_raphson::NewtonRaphson;

use crate::{
    defeat_message,
    math::{
        TestError,
        integrate::IntegrationError,
        matrix::square::{Banded, SquareMatrixError},
    },
};
use std::fmt::{self, Debug, Display, Formatter};

/// Zeroth-order root-finding algorithms.
pub trait ZerothOrderRootFinding<X> {
    fn root(
        &self,
        function: impl Fn(&X) -> Result<X, OptimizeError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizeError>;
}

/// First-order root-finding algorithms.
pub trait FirstOrderRootFinding<F, J, X> {
    fn root(
        &self,
        function: impl Fn(&X) -> Result<F, OptimizeError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizeError>;
}

/// First-order optimization algorithms.
pub trait FirstOrderOptimization<F, X> {
    fn minimize(
        &self,
        function: impl Fn(&X) -> Result<F, OptimizeError>,
        jacobian: impl Fn(&X) -> Result<X, OptimizeError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizeError>;
}

/// Second-order optimization algorithms.
pub trait SecondOrderOptimization<F, J, H, X> {
    fn minimize(
        &self,
        function: impl Fn(&X) -> Result<F, OptimizeError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        hessian: impl Fn(&X) -> Result<H, OptimizeError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
        banded: Option<Banded>,
    ) -> Result<X, OptimizeError>;
}

/// Possible errors encountered during optimization.
pub enum OptimizeError {
    Generic(String),
    MaximumStepsReached(usize, String),
    NotMinimum(String, String),
    SingularMatrix,
}

impl From<OptimizeError> for String {
    fn from(error: OptimizeError) -> Self {
        error.to_string()
    }
}

impl From<OptimizeError> for TestError {
    fn from(error: OptimizeError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl From<IntegrationError> for OptimizeError {
    fn from(_error: IntegrationError) -> Self {
        todo!()
    }
}

impl From<SquareMatrixError> for OptimizeError {
    fn from(_error: SquareMatrixError) -> Self {
        Self::SingularMatrix
    }
}

impl Debug for OptimizeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Generic(message) => message.to_string(),
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

impl Display for OptimizeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Generic(message) => message.to_string(),
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
