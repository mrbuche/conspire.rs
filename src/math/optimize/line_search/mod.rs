mod armijo;
mod error;
mod goldstein;
mod wolfe;

use super::{super::Scalar, OptimizeError};
use crate::math::{Jacobian, Solution};
use std::ops::Mul;

/// Available line search algorithms.
#[derive(Debug)]
pub enum LineSearch {
    /// The Armijo condition.
    Armijo(Scalar, Scalar, usize),
    /// Backtrack for errors.
    Error(Scalar, usize),
    /// The Goldstein conditions.
    Goldstein(Scalar, Scalar, usize),
    /// The Wolfe conditions.
    Wolfe(Scalar, Scalar, Scalar, usize, bool),
    /// No line search.
    None,
}

impl Default for LineSearch {
    fn default() -> Self {
        Self::Armijo(1e-3, 9e-1, 100)
    }
}

impl LineSearch {
    pub fn backtrack<X, J>(
        &self,
        function: impl Fn(&X) -> Result<Scalar, OptimizeError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        argument: &X,
        jacobian0: &J,
        decrement: &X,
        step_size: &Scalar,
    ) -> Result<Scalar, OptimizeError>
    where
        J: Jacobian,
        for<'a> &'a J: From<&'a X>,
        X: Solution,
        for<'a> &'a X: Mul<Scalar, Output = X>,
    {
        match self {
            Self::Armijo(control, cut_back, max_steps) => armijo::backtrack(
                *control, *cut_back, *max_steps, function, jacobian0, argument, decrement,
                step_size,
            ),
            Self::Error(cut_back, max_steps) => error::backtrack(
                *cut_back, *max_steps, function, jacobian0, argument, decrement, step_size,
            ),
            Self::Goldstein(control, cut_back, max_steps) => goldstein::backtrack(
                *control, *cut_back, *max_steps, function, jacobian0, argument, decrement,
                step_size,
            ),
            Self::Wolfe(control_1, control_2, cut_back, max_steps, strong) => wolfe::backtrack(
                *control_1, *control_2, *cut_back, *max_steps, *strong, function, jacobian,
                argument, jacobian0, decrement, step_size,
            ),
            Self::None => {
                panic!("Cannot call backtracking line search when there is no algorithm.")
            }
        }
    }
}

// /// Possible errors encountered during line search.
// pub enum OptimizeError {
//     MaximumStepsReached(usize, String),
//     NegativeStepSize(String, String),
//     NotDescentDirection(String, String),
// }
// Need to implement conversion to OptimizeError and so on.
