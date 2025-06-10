mod armijo;

use armijo::Armijo;

use super::{super::Scalar, OptimizeError};
use crate::math::{Jacobian, Solution};
use std::ops::Mul;

/// Required methods for line search algorithms.
pub trait Search {
    /// Perform a line search calculation.
    fn line_search<X, J>(
        &self,
        function: impl Fn(&X) -> Result<Scalar, OptimizeError>,
        jacobian: &J,
        argument: &X,
        decrement: &X,
        step_size: &Scalar,
    ) -> Result<Scalar, OptimizeError>
    where
        J: Jacobian,
        for<'a> &'a J: From<&'a X>,
        X: Solution,
        for<'a> &'a X: Mul<Scalar, Output = X>;
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

impl Search for LineSearch {
    fn line_search<X, J>(
        &self,
        function: impl Fn(&X) -> Result<Scalar, OptimizeError>,
        jacobian: &J,
        argument: &X,
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
            Self::Armijo(control, cut_back, max_steps) => Armijo {
                control: *control,
                cut_back: *cut_back,
                max_steps: *max_steps,
            },
        }
        .line_search(function, jacobian, argument, decrement, step_size)
    }
}
