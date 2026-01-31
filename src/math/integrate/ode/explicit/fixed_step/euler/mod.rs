#[cfg(test)]
mod test;

use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::{Explicit, FixedStep, FixedStepExplicit, IntegrationError, OdeSolver},
};
use std::ops::Mul;

#[doc = include_str!("doc.md")]
#[derive(Debug, Default)]
pub struct Euler {
    /// Fixed value for the time step.
    dt: Scalar,
}

impl<Y, U> OdeSolver<Y, U> for Euler
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl FixedStep for Euler {
    fn dt(&self) -> Scalar {
        self.dt
    }
}

impl<Y, U> Explicit<Y, U> for Euler
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y>,
    U: TensorVec<Item = Y>,
{
    const SLOPES: usize = 1;
    fn integrate(
        &self,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        time: &[Scalar],
        initial_condition: Y,
    ) -> Result<(Vector, U, U), IntegrationError> {
        self.integrate_fixed_step(function, time, initial_condition)
    }
}

impl<Y, U> FixedStepExplicit<Y, U> for Euler
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn step(
        &self,
        mut function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &Y,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
    ) -> Result<(), String> {
        k[0] = function(t, y)?;
        *y_trial = &k[0] * dt + y;
        Ok(())
    }
}
