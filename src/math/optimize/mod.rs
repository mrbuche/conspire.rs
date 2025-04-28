#[cfg(test)]
mod test;

mod constraint;
mod gradient_descent;
mod newton_raphson;

use super::{Hessian, Tensor, TensorRank0};
use crate::defeat_message;
use std::{fmt, ops::{Div, SubAssign}};

pub use constraint::{IntoConstraint, equality::{EqualityConstraint, linear::LinearEqualityConstraint}};
pub use gradient_descent::GradientDescent;
pub use newton_raphson::NewtonRaphson;

/// Dirichlet boundary conditions.
pub struct Dirichlet {
    pub places: Vec<usize>,
    pub values: Vec<TensorRank0>,
}

/// First-order optimization algorithms.
pub trait FirstOrder<X: Tensor> {
    fn minimize(
        &self,
        jacobian: impl Fn(&X) -> Result<X, OptimizeError>,
        initial_guess: X,
        dirichlet: Option<Dirichlet>,
    ) -> Result<X, OptimizeError>;
}

/// Second-order optimization algorithms.
pub trait SecondOrder<C, H, J, X>
where
    C: IntoConstraint<LinearEqualityConstraint>,
    H: Hessian,
    J: Div<H, Output = X> + Tensor,
    X: Tensor,
{
    fn minimize(
        &self,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        hessian: impl Fn(&X) -> Result<H, OptimizeError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint<C>,
    ) -> Result<X, OptimizeError>;
}

// maybe put below in another directory for root finding?
// also would be a first-order root finding method?
// how does scipy organize this?

/// Second-order solution algorithms.
pub trait SecondOrderRoot<H, J, X>
where
    H: Hessian,
    J: Div<H, Output = X> + Tensor,
    X: Tensor,
{
    fn root(
        &self,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        hessian: impl Fn(&X) -> Result<H, OptimizeError>,
        initial_guess: X,
    ) -> Result<X, OptimizeError>;
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
