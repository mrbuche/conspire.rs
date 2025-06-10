use super::{OptimizeError, Scalar, Search};
use crate::math::{Jacobian, Solution};
use std::ops::Mul;

/// The Armijo condition.
#[derive(Debug)]
pub struct Armijo {
    /// Control parameter.
    pub control: Scalar,
    /// Cut-back parameter.
    pub cut_back: Scalar,
    /// Maximum number of steps.
    pub max_steps: usize,
}

impl Default for Armijo {
    fn default() -> Self {
        Self {
            control: 1e-3,
            cut_back: 0.9,
            max_steps: 25,
        }
    }
}

impl Search for Armijo {
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
        assert!(step_size > &0.0, "Negative step size");
        let mut n = -step_size;
        let f = function(argument)?;
        let m = jacobian.full_contraction(decrement.into());
        assert!(m > 0.0, "Not a descent direction");
        let t = self.control * m;
        for _ in 0..self.max_steps {
            if function(&(decrement * n + argument))? - f > n * t {
                n *= self.cut_back
            } else {
                return Ok(-n);
            }
        }
        Err(OptimizeError::MaximumStepsReached(
            self.max_steps,
            format!("{:?}", self),
        ))
    }
}
