use super::{OptimizeError, Scalar};
use crate::math::{Jacobian, Solution};
use std::ops::Mul;

pub fn backtrack<X, J>(
    control: Scalar,
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
    let f = function(argument)?;
    let m = jacobian(argument)?.full_contraction(decrement.into());
    assert!(m > 0.0, "Not a descent direction");
    let t = control * m;
    let u = (1.0 - control) * m;
    let mut v;
    for _ in 0..max_steps {
        v = function(&(decrement * n + argument))? - f;
        if n * u > v || v > n * t {
            n *= cut_back
        } else {
            return Ok(-n);
        }
    }
    panic!("Maximum steps reached")
}
