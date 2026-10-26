use super::{LineSearchError, ObjectiveMerit, ResidualMerit};
use crate::{
    math::{Quantity, Scalar},
    units::UnitMul,
};
use std::fmt::{self, Display, Formatter};

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
    fn search<W, U, V>(
        &self,
        _value: impl FnMut(Quantity<W>) -> Result<Quantity<V>, String>,
        _slope: impl FnMut(Quantity<W>) -> Result<Quantity<U>, String>,
        step_size: Quantity<W>,
    ) -> Result<Quantity<W>, LineSearchError>
    where
        W: UnitMul<U, Output = V>,
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
