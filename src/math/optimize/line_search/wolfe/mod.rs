use super::{OptimizeError, Scalar};
use crate::math::{Jacobian, Solution};
use std::ops::Mul;

pub fn backtrack<X, J>(
    control_1: Scalar,
    control_2: Scalar,
    cut_back: Scalar,
    max_steps: usize,
    strong: bool,
    function: impl Fn(&X) -> Result<Scalar, OptimizeError>,
    jacobian: impl Fn(&X) -> Result<J, OptimizeError>,
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
    todo!()
}
