#[cfg(test)]
mod test;

mod constraint;
mod gradient_descent;
mod newton_raphson;

use crate::defeat_message;
use std::fmt::{self, Debug, Display, Formatter};

pub use constraint::EqualityConstraint;
pub use gradient_descent::GradientDescent;
pub use newton_raphson::NewtonRaphson;

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
pub trait SecondOrderOptimization<F, H, J, X> {
    fn minimize(
        &self,
        function: impl Fn(&X) -> Result<F, OptimizeError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        hessian: impl Fn(&X) -> Result<H, OptimizeError>,
        initial_guess: X,
        equality_constraint: EqualityConstraint,
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

impl Debug for OptimizeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
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

impl Display for OptimizeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
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
