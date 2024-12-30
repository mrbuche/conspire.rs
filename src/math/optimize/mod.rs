#[cfg(test)]
mod test;

mod gradient_descent;
mod newton_raphson;

use super::{Hessian, Tensor, TensorRank0};
use crate::get_defeat_message;
use std::{fmt, ops::Div};

pub use gradient_descent::GradientDescent;
pub use newton_raphson::NewtonRaphson;

/// Dirichlet boundary conditions.
pub struct Dirichlet<'a> {
    pub places: &'a [&'a [usize]],
    pub values: &'a [TensorRank0],
}

/// Neumann boundary conditions.
pub struct Neumann<'a> {
    pub places: &'a [&'a [usize]],
    pub values: &'a [TensorRank0],
}

/// First-order optimization algorithms.
pub trait FirstOrder<X: Tensor> {
    fn minimize(
        &self,
        jacobian: impl Fn(&X) -> Result<X, OptimizeError>,
        initial_guess: X,
        dirichlet: Option<Dirichlet>,
        neumann: Option<Neumann>,
    ) -> Result<X, OptimizeError>;
}

/// Second-order optimization algorithms.
pub trait SecondOrder<H: Hessian, J: Tensor, X: Tensor>
where
    J: Div<H, Output = X>,
{
    fn minimize(
        &self,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        hessian: impl Fn(&X) -> Result<H, OptimizeError>,
        initial_guess: X,
        dirichlet: Option<Dirichlet>,
        neumann: Option<Neumann>,
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
        write!(
            f,
            "\n{}\n\x1b[0;2;31m{}\x1b[0m\n",
            error,
            get_defeat_message()
        )
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
