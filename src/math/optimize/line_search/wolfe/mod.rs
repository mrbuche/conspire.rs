use super::{OptimizeError, Scalar};
use crate::math::{Jacobian, Solution};
use std::ops::Mul;

#[allow(clippy::too_many_arguments)]
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
    assert!(step_size > &0.0, "Negative step size");
    let mut n = -step_size;
    let f = function(argument)?;
    let m = jacobian(argument)?.full_contraction(decrement.into());
    assert!(m > 0.0, "Not a descent direction");
    let t_1 = control_1 * m;
    let t_2 = control_2 * m;
    let mut trial_argument = decrement * n + argument;
    for _ in 0..max_steps {
        if function(&trial_argument)? - f > n * t_1
            || if strong {
                jacobian(&trial_argument)?.full_contraction(decrement.into()) > t_2
            } else {
                jacobian(&trial_argument)?
                    .full_contraction(decrement.into())
                    .abs()
                    > t_2.abs()
            }
        {
            n *= cut_back;
            trial_argument = decrement * n + argument
        } else {
            return Ok(-n);
        }
    }
    panic!("Maximum steps reached")
}
