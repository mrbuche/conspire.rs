use super::{OptimizeError, Scalar};
use crate::math::{Jacobian, Solution};
use std::ops::Mul;

#[allow(clippy::too_many_arguments)]
pub fn backtrack<X, J>(
    control: Scalar,
    cut_back: Scalar,
    max_steps: usize,
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
    assert!(step_size > &0.0, "Negative step size");
    let mut f_n;
    let mut n = -step_size;
    let f = function(argument)?;
    let m = jacobian.full_contraction(decrement.into());
    assert!(m > 0.0, "Not a descent direction");
    let t = control * m;
    for _ in 0..max_steps {
        f_n = function(&(decrement * n + argument));
        if f_n.is_err() || f_n? - f > n * t {
            n *= cut_back
        } else {
            return Ok(-n);
        }
    }
    panic!("Maximum steps reached")
}
