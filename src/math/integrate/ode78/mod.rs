#[cfg(test)]
mod test;

use super::{
    super::{
        interpolate::InterpolateSolution, Tensor, TensorArray, TensorRank0, TensorVec, Vector,
    },
    Explicit, IntegrationError,
};
use crate::{ABS_TOL, REL_TOL};
use std::ops::{Mul, Sub};

/// ???
#[derive(Debug)]
pub struct Ode78 {
    /// Absolute error tolerance.
    pub abs_tol: TensorRank0,
    /// Multiplying factor when decreasing time steps.
    pub dec_fac: TensorRank0,
    /// Initial relative timestep.
    pub dt_init: TensorRank0,
    /// Multiplying factor when increasing time steps.
    pub inc_fac: TensorRank0,
    /// Relative error tolerance.
    pub rel_tol: TensorRank0,
}

impl Default for Ode78 {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            dec_fac: 0.5,
            dt_init: 0.1,
            inc_fac: 1.1,
            rel_tol: REL_TOL,
        }
    }
}

impl<Y, U> Explicit<Y, U> for Ode78
where
    Self: InterpolateSolution<Y, U>,
    Y: Tensor + TensorArray,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn integrate(
        &self,
        function: impl Fn(&TensorRank0, &Y) -> Y,
        initial_time: TensorRank0,
        initial_condition: Y,
        time: &[TensorRank0],
    ) -> Result<(Vector, U), IntegrationError> {
        todo!()
    }
}

impl<Y, U> InterpolateSolution<Y, U> for Ode78
where
    Y: Tensor + TensorArray,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn interpolate(
        &self,
        ti: &Vector,
        tp: &Vector,
        yp: &U,
        f: impl Fn(&TensorRank0, &Y) -> Y,
    ) -> U {
        todo!()
    }
}
