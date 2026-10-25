use super::{LineSearchError, ObjectiveMerit, ResidualMerit, backtrack, exhausted};
use crate::math::{Erase, Jacobian, Quantity, Scalar, Solution, Tensor};
use std::{
    fmt::{self, Display, Formatter},
    ops::Mul,
};

/// Backtrack for errors.
#[derive(Clone, Debug)]
pub struct Error {
    pub cut_back: Scalar,
    pub max_steps: usize,
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Error")
    }
}

impl Error {
    fn reachable(
        &self,
        mut merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        step_size: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        let mut n = step_size;
        for _ in 0..self.max_steps {
            if merit(n).is_ok() {
                return Ok(n);
            } else {
                n *= self.cut_back
            }
        }
        Err(exhausted(self, self.max_steps))
    }
}

impl ObjectiveMerit for Error {
    const MERIT: bool = false;
    fn search<X, J, D, W, E>(
        &self,
        function: impl FnMut(&X, Scalar) -> Result<Scalar, String>,
        _jacobian: impl FnMut(&X) -> Result<J, String>,
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
        backtrack(
            self,
            self.cut_back,
            self.max_steps,
            function,
            argument,
            jacobian0,
            decrement,
            step_size,
            |_, _, _| true,
        )
    }
    fn search_merit(
        &self,
        merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        _value: Scalar,
        _slope: Scalar,
        step_size: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        self.reachable(merit, step_size)
    }
}

impl ResidualMerit for Error {
    const MERIT: bool = false;
    fn search_residual(
        &self,
        merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        _value: Scalar,
        _fraction: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        self.reachable(merit, 1.0)
    }
}
