use super::{OptimizeError, Scalar};
use crate::math::{Jacobian, Solution};
use std::ops::Mul;

#[allow(clippy::too_many_arguments)]
pub fn backtrack<X, J>(
    cut_back: Scalar,
    max_steps: usize,
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
    assert!(step_size > &0.0, "Negative step size");
    let mut n = -step_size;
    let m = jacobian(argument)?.full_contraction(decrement.into());
    assert!(m > 0.0, "Not a descent direction");
    for _ in 0..max_steps {
        if function(&(decrement * n + argument)).is_err() {
            n *= cut_back
        } else {
            return Ok(-n);
        }
    }
    panic!("Maximum steps reached")
}
