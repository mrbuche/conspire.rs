#[cfg(test)]
mod test;

mod armijo;

use armijo::Armijo;

use super::{super::TensorRank0, OptimizeError};
use crate::math::{Jacobian, Tensor};
use std::ops::Mul;

/// ???
pub trait Search<F, J, X> {
    /// ???
    fn line_search(
        &self,
        function: impl Fn(&X) -> Result<F, OptimizeError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        position: &X,
        direction: &X,
        step_size: &mut F,
    ) -> Result<(), OptimizeError>;
}

/// ???
#[derive(Debug)]
pub enum LineSearch {
    Armijo(TensorRank0, TensorRank0, usize),
}

impl<F, J, X> Search<F, J, X> for LineSearch
where
    // J: Jacobian + for<'a> Mul<&'a X, Output = F>,
    // for<'a> &'a J: Mul<F, Output = X>,
    X: Tensor,
{
    fn line_search(
        &self,
        function: impl Fn(&X) -> Result<F, OptimizeError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
        position: &X,
        direction: &X,
        step_size: &mut F,
    ) -> Result<(), OptimizeError> {
        match self {
            Self::Armijo(control, cut_back, max_steps) => Armijo {
                control: *control,
                cut_back: *cut_back,
                max_steps: *max_steps,
            }
            .line_search(function, jacobian, position, direction, step_size),
        }
    }
}
