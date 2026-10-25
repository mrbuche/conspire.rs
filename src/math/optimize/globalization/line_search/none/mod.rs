use super::{LineSearchError, ObjectiveMerit, ResidualMerit};
use crate::math::{Erase, Jacobian, Quantity, Scalar, Solution, Tensor};
use std::{
    fmt::{self, Display, Formatter},
    ops::Mul,
};

/// No line search.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoSearch;

impl Display for NoSearch {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "NoSearch")
    }
}

impl ObjectiveMerit for NoSearch {
    const SEARCHES: bool = false;
    const MERIT: bool = false;
    fn search<X, J, D, W, E>(
        &self,
        _function: impl FnMut(&X, Scalar) -> Result<Scalar, String>,
        _jacobian: impl FnMut(&X) -> Result<J, String>,
        _argument: &X,
        _jacobian0: &J,
        _decrement: &D,
        step_size: Scalar,
    ) -> Result<Scalar, LineSearchError>
    where
        J: Erase<Erased = E> + Jacobian,
        D: Erase<Erased = E>,
        E: Tensor,
        X: Solution,
        for<'a> &'a D: Mul<Quantity<W>, Output = X>,
    {
        Ok(step_size)
    }
    fn search_merit(
        &self,
        _merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        _value: Scalar,
        _slope: Scalar,
        step_size: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        Ok(step_size)
    }
}

impl ResidualMerit for NoSearch {
    const SEARCHES: bool = false;
    const MERIT: bool = false;
    fn search_residual(
        &self,
        _merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        _value: Scalar,
        _fraction: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        Ok(1.0)
    }
}
