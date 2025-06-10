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
            control: 1e-3,
            cut_back: 0.9,
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

        assert!(step_size > &0.0);

        let mut a = *step_size;
        let f = function(position)?;
        let m = jacobian(position)?.full_contraction(direction.into());
        //
        // direction is coming in with sign flipped right now
        //
        assert!(m > 0.0);
        //
        // so putting a negative sign below on m and on a in function call
        //
        let t = self.control * -m;
        for _ in 0..self.max_steps {
            if function(&(direction * -a + position))? - f > a * t {
                println!("{:?}, {:?}, {:?}", a, a*t, function(&(direction * -a + position))? - f);
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
