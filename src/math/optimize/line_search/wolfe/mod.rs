use super::{LineSearchError, ObjectiveMerit, exhausted, start};
use crate::math::{Erase, Jacobian, Quantity, Scalar, Solution, Tensor};
use std::{
    fmt::{self, Display, Formatter},
    ops::Mul,
};

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
    fn search<X, J, D, W, E>(
        &self,
        mut function: impl FnMut(&X, Scalar) -> Result<Scalar, String>,
        mut jacobian: impl FnMut(&X) -> Result<J, String>,
        argument: &X,
        jacobian0: &J,
        decrement: &D,
        step_size: Scalar,
    ) -> Result<Scalar, LineSearchError>
    where
        J: Erase<Erased = E> + Jacobian,
        D: Erase<Erased = E>,
        E: Tensor,
        X: Solution,
        for<'a> &'a D: Mul<Quantity<W>, Output = X>,
    {
        let (value, slope) = start(
            self,
            &mut function,
            argument,
            jacobian0,
            decrement,
            step_size,
        )?;
        let t_1 = self.control_1 * slope;
        let t_2 = self.control_2 * slope;
        let mut n = step_size;
        for _ in 0..self.max_steps {
            let trial = decrement * Quantity::new(-n) + argument;
            if let Ok(trial_value) = function(&trial, n)
                && let Ok(trial_jacobian) = jacobian(&trial)
                && value - trial_value >= n * t_1
                && {
                    let curvature = trial_jacobian.erase().full_contraction(decrement.erase());
                    if self.strong {
                        curvature.abs() < t_2.abs()
                    } else {
                        curvature < t_2
                    }
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
