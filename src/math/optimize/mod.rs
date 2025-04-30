#[cfg(test)]
mod test;

mod constraint;
mod gradient_descent;
mod newton_raphson;

use super::{Hessian, SquareMatrix, Tensor, TensorRank0, Vector};
use crate::defeat_message;
use std::{fmt, ops::Div};

pub use constraint::EqualityConstraint;
pub use gradient_descent::GradientDescent;
pub use newton_raphson::NewtonRaphson;

/// Dirichlet boundary conditions.
pub struct Dirichlet {
    pub places: Vec<usize>,
    pub values: Vec<TensorRank0>,
}

/// Zeroth-order root-finding algorithms.
pub trait ZerothOrderRootFinding<X>
where
    X: Tensor,
{
    fn root(
        &self,
        function: impl Fn(&X) -> Result<X, OptimizeError>,
        initial_guess: X,
    ) -> Result<X, OptimizeError>;
    fn solve(
        &self,
        function: impl Fn(&X) -> Result<X, OptimizeError>,
        initial_guess: X,
        dirichlet: Option<Dirichlet>,
    ) -> Result<X, OptimizeError>;
}

/// First-order root-finding algorithms.
pub trait FirstOrderRootFinding {
    fn root<F, J, X>(
        &self,
        function: impl Fn(&X) -> Result<F, OptimizeError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        initial_guess: X,
    ) -> Result<X, OptimizeError>
    where
        F: Div<J, Output = X> + Tensor,
        X: Tensor;
    fn solve(
        &self,
        function: impl Fn(&Vector) -> Result<Vector, OptimizeError>,
        jacobian: impl Fn(&Vector) -> Result<SquareMatrix, OptimizeError>,
        initial_guess: Vector,
        equality_constraint: EqualityConstraint,
    ) -> Result<Vector, OptimizeError>;
}

/// First-order optimization algorithms.
pub trait FirstOrderOptimization<F, X>
where
    X: Tensor,
{
    fn minimize(
        &self,
        function: impl Fn(&X) -> Result<F, OptimizeError>,
        jacobian: impl Fn(&X) -> Result<X, OptimizeError>,
        initial_guess: X,
    ) -> Result<X, OptimizeError>;
}

/// Second-order optimization algorithms.
pub trait SecondOrderOptimization {
    fn minimize<F, H, J, X>(
        &self,
        function: impl Fn(&X) -> Result<F, OptimizeError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        hessian: impl Fn(&X) -> Result<H, OptimizeError>,
        initial_guess: X,
    ) -> Result<X, OptimizeError>
    where
        H: Hessian,
        J: Div<H, Output = X> + Tensor,
        X: Tensor;
    fn minimize_constrained(
        &self,
        function: impl Fn(&Vector) -> Result<TensorRank0, OptimizeError>,
        jacobian: impl Fn(&Vector) -> Result<Vector, OptimizeError>,
        hessian: impl Fn(&Vector) -> Result<SquareMatrix, OptimizeError>,
        initial_guess: Vector,
        equality_constraint: EqualityConstraint,
    ) -> Result<Vector, OptimizeError>;
}

/// Possible optimization algorithms.
#[derive(Debug)]
pub enum Optimization {
    GradientDescent(GradientDescent),
    NewtonRaphson(NewtonRaphson),
}

/// Possible errors encountered when optimizing.
pub enum OptimizeError {
    MaximumStepsReached(usize, String),
    NotMinimum(String, String),
}

impl fmt::Debug for OptimizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::MaximumStepsReached(steps, optimizer) => {
                format!(
                    "\x1b[1;91mMaximum number of steps ({}) reached.\x1b[0;91m\n\
                     In optimizer: {}.",
                    steps, optimizer
                )
            }
            Self::NotMinimum(solution, optimizer) => {
                format!(
                    "\x1b[1;91mThe obtained solution is not a minimum.\x1b[0;91m\n\
                     For solution: {}.\n\
                     In optimizer: {}.",
                    solution, optimizer
                )
            }
        };
        write!(f, "\n{}\n\x1b[0;2;31m{}\x1b[0m\n", error, defeat_message())
    }
}

impl fmt::Display for OptimizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::MaximumStepsReached(steps, optimizer) => {
                format!(
                    "\x1b[1;91mMaximum number of steps ({}) reached.\x1b[0;91m\n\
                     In optimizer: {}.",
                    steps, optimizer
                )
            }
            Self::NotMinimum(solution, optimizer) => {
                format!(
                    "\x1b[1;91mThe obtained solution is not a minimum.\x1b[0;91m\n\
                     For solution: {}.\n\
                     In optimizer: {}.",
                    solution, optimizer
                )
            }
        };
        write!(f, "{}\x1b[0m", error)
    }
}
