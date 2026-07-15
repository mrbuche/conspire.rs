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

use crate::math::{
    Jacobian, Scalar, Solution, Style, StyledError,
    assert::AssertionError,
    matrix::square::SquareMatrixError,
    sparse::{SparseError, SparseSolver},
    styled_error,
};
use std::{fmt::Debug, ops::Mul};

/// Zeroth-order root-finding algorithms.
pub trait ZerothOrderRootFinding<X> {
    fn root(
        &self,
        function: impl FnMut(&X) -> Result<X, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizationError>;
}

/// First-order root-finding algorithms.
pub trait FirstOrderRootFinding<F, J, X> {
    fn root(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
        sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError>;
}

/// First-order optimization algorithms.
pub trait FirstOrderOptimization<F, X> {
    fn minimize(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<X, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizationError>;
}

/// Second-order optimization algorithms.
pub trait SecondOrderOptimization<F, J, H, X> {
    fn minimize(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        hessian: impl FnMut(&X) -> Result<H, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
        sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError>;
}

trait BacktrackingLineSearch<J, X>
where
    Self: Debug,
{
    fn backtracking_line_search(
        &self,
        mut function: impl FnMut(&X) -> Result<Scalar, String>,
        mut jacobian: impl FnMut(&X) -> Result<J, String>,
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
                &mut function,
                &mut jacobian,
                argument,
                jacobian0,
                decrement,
                step_size,
            ) {
                Ok(step_size) => Ok(step_size),
                Err(error) => Err(OptimizationError::Upstream(
                    format!("{error}"),
                    format!("{self:?}"),
                )),
            }
        }
    }
    fn get_line_search(&self) -> &LineSearch;
}

/// Possible errors encountered during optimization.
pub enum OptimizationError {
    Intermediate(String),
    MaximumStepsReached(usize, String),
    NotMinimum(String, String),
    Upstream(String, String),
    SingularMatrix,
}

impl From<String> for OptimizationError {
    fn from(error: String) -> Self {
        Self::Intermediate(error)
    }
}

impl StyledError for OptimizationError {
    fn message(&self, style: &Style) -> String {
        let (h, c) = (style.headline, style.frame);
        match self {
            Self::Intermediate(message) => message.to_string(),
            Self::MaximumStepsReached(steps, solver) => format!(
                "{h}Maximum number of steps ({steps}) reached.{c}\n\
                In solver: {solver}."
            ),
            Self::NotMinimum(solution, solver) => format!(
                "{h}The obtained solution is not a minimum.{c}\n\
                For solution: {solution}.\n\
                In solver: {solver}."
            ),
            Self::SingularMatrix => format!("{h}Matrix is singular."),
            Self::Upstream(error, solver) => format!(
                "{error}{c}\n\
                In solver: {solver}."
            ),
        }
    }
}

styled_error!(OptimizationError);

impl From<OptimizationError> for String {
    fn from(error: OptimizationError) -> Self {
        error.to_string()
    }
}

impl From<OptimizationError> for AssertionError {
    fn from(error: OptimizationError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl From<SquareMatrixError> for OptimizationError {
    fn from(_error: SquareMatrixError) -> Self {
        Self::SingularMatrix
    }
}

impl From<SparseError> for OptimizationError {
    fn from(_error: SparseError) -> Self {
        Self::SingularMatrix
    }
}
