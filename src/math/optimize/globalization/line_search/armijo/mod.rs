use super::{LineSearchError, ObjectiveMerit, backtrack, backtrack_merit};
use crate::{
    math::{Quantity, Scalar},
    units::UnitMul,
};
use std::{
    fmt::{self, Display, Formatter},
    ops::Mul,
};

/// The Armijo condition on the objective.
#[derive(Clone, Debug)]
pub struct Armijo {
    pub control: Scalar,
    pub cut_back: Scalar,
    pub max_steps: usize,
}

impl Default for Armijo {
    fn default() -> Self {
        Self {
            control: 1e-3,
            cut_back: 9e-1,
            max_steps: 100,
        }
    }
}

impl Display for Armijo {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Armijo")
    }
}

impl Armijo {
    fn accepts<T>(&self, decrease: T, n_slope: T) -> bool
    where
        T: PartialOrd + Mul<Scalar, Output = T>,
    {
        decrease >= n_slope * self.control
    }
}

impl ObjectiveMerit for Armijo {
    fn search<W, U, V>(
        &self,
        value: impl FnMut(Quantity<W>) -> Result<Quantity<V>, String>,
        slope: impl FnMut(Quantity<W>) -> Result<Quantity<U>, String>,
        step_size: Quantity<W>,
    ) -> Result<Quantity<W>, LineSearchError<W>>
    where
        W: UnitMul<U, Output = V>,
    {
        backtrack(
            self,
            self.cut_back,
            self.max_steps,
            value,
            slope,
            step_size,
            |decrease, n_slope| self.accepts(decrease, n_slope),
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
            |decrease, n_slope| self.accepts(decrease, n_slope),
        )
    }
}
