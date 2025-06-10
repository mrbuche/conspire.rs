#[cfg(test)]
mod test;

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
            control: 0.5,
            cut_back: 0.5,
            max_steps: 10,
        }
    }
}

impl<J, X> Search<Scalar, J, X> for Armijo
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
        // assert!(step_size > 0.0);
        let mut a = *step_size;
        let f = function(position)?;
        let m = jacobian(position)?.full_contraction(direction.into());
        // assert!(m < 0.0);
        let t = self.control * m;
        for _ in 0..self.max_steps {
            if function(&(direction * a + position))? - f > a * t {
                a *= self.cut_back
            } else {
                return Ok(a);
            }
        }
        Err(OptimizeError::MaximumStepsReached(
            self.max_steps,
            format!("{:?}", self),
        ))
    }
}
