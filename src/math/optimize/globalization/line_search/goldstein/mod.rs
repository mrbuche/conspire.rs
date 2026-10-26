use super::{LineSearchError, ObjectiveMerit, backtrack, backtrack_merit};
use crate::math::{Erase, Jacobian, Quantity, Scalar, Solution, Tensor};
use std::{
    fmt::{self, Display, Formatter},
    ops::Mul,
};

/// The Goldstein conditions on the objective.
#[derive(Clone, Debug)]
pub struct Goldstein {
    pub control: Scalar,
    pub cut_back: Scalar,
    pub max_steps: usize,
}

impl Display for Goldstein {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Goldstein")
    }
}

impl Goldstein {
    fn accepts(&self, decrease: Scalar, n: Scalar, slope: Scalar) -> bool {
        !(n * (1.0 - self.control) * slope < decrease || decrease < n * self.control * slope)
    }
}

impl ObjectiveMerit for Goldstein {
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
            |decrease, n, slope| self.accepts(decrease, n, slope),
        )
    }
    fn search_merit(
        &self,
        merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        value: Scalar,
        slope: Scalar,
        step_size: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        backtrack_merit(
            self,
            self.cut_back,
            self.max_steps,
            merit,
            value,
            slope,
            step_size,
            |decrease, n, slope| self.accepts(decrease, n, slope),
        )
    }
}
