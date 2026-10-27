use super::{LineSearchError, ObjectiveMerit, exhausted, start};
use crate::{
    math::{Quantity, Scalar},
    units::UnitMul,
};
use std::fmt::{self, Display, Formatter};

/// The Wolfe conditions on the objective.
#[derive(Clone, Debug)]
pub struct Wolfe {
    pub control_1: Scalar,
    pub control_2: Scalar,
    pub cut_back: Scalar,
    pub max_steps: usize,
    pub strong: bool,
}

impl Display for Wolfe {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Wolfe")
    }
}

impl ObjectiveMerit for Wolfe {
    fn search<W, U, V>(
        &self,
        mut value: impl FnMut(Quantity<W>) -> Result<Quantity<V>, String>,
        mut slope: impl FnMut(Quantity<W>) -> Result<Quantity<U>, String>,
        step_size: Quantity<W>,
    ) -> Result<Quantity<W>, LineSearchError<W>>
    where
        W: UnitMul<U, Output = V>,
    {
        let (initial, gradient) = start(self, &mut value, &mut slope, step_size)?;
        let t_1 = gradient * self.control_1;
        let t_2 = gradient * self.control_2;
        let mut n = step_size;
        for _ in 0..self.max_steps {
            if let Ok(trial) = value(n)
                && let Ok(curvature) = slope(n)
                && initial - trial >= n * t_1
                && if self.strong {
                    curvature.abs() < t_2.abs()
                } else {
                    curvature < t_2
                }
            {
                return Ok(n);
            } else {
                n *= self.cut_back
            }
        }
        Err(exhausted(self, self.max_steps))
    }
    fn search_merit(
        &self,
        _merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        _value: Scalar,
        _slope: Scalar,
        _step_size: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        panic!(
            "The Wolfe conditions need the gradient of the merit function, which the exact penalty function does not have."
        )
    }
}
