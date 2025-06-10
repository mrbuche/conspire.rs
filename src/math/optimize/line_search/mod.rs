#[cfg(test)]
mod test;

mod armijo;

use armijo::Armijo;

use super::{super::Scalar, OptimizeError};
use crate::math::{Jacobian, Solution};
use std::ops::Mul;

/// Required methods for line search algorithms.
pub trait Search<F, J, X> {
    /// Perform a line search calculation.
    fn line_search(
        &self,
        function: impl Fn(&X) -> Result<F, OptimizeError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        position: &X,
        direction: &X,
        step_size: &F,
    ) -> Result<F, OptimizeError>;
}

/// Possible line search algorithms.
#[derive(Debug)]
pub enum LineSearch {
    Armijo(Scalar, Scalar, usize),
}

impl Default for LineSearch {
    fn default() -> Self {
        let default = Armijo::default();
        Self::Armijo(default.control, default.cut_back, default.max_steps)
    }
}

impl<J, X> Search<Scalar, J, X> for LineSearch
where
    J: Jacobian,
    for<'a> &'a J: From<&'a X>,
    X: Solution,
    for<'a> &'a X: Mul<Scalar, Output = X>,
{
    fn line_search(
        &self,
        function: impl Fn(&X) -> Result<Scalar, OptimizeError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        position: &X,
        direction: &X,
        step_size: &Scalar,
    ) -> Result<Scalar, OptimizeError> {
        match self {
            Self::Armijo(control, cut_back, max_steps) => Armijo {
                control: *control,
                cut_back: *cut_back,
                max_steps: *max_steps,
            },
        }
        .line_search(function, jacobian, position, direction, step_size)
    }
}
