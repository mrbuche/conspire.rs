#[cfg(test)]
mod test;

use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::{Explicit, FixedStep, FixedStepExplicit, IntegrationError, OdeIntegrator},
};
use std::ops::{Add, Mul};

#[doc = include_str!("doc.md")]
#[derive(Debug, Default)]
pub struct Ralston {
    /// Fixed value for the time step.
    dt: Scalar,
}

impl<Y, U> OdeIntegrator<Y, U> for Ralston
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl FixedStep for Ralston {
    fn dt(&self) -> Scalar {
        self.dt
    }
}

impl<Y, U> Explicit<Y, U> for Ralston
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Add<Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    const SLOPES: usize = 2;
    fn integrate(
        &self,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        time: &[Scalar],
        initial_condition: Y,
    ) -> Result<(Vector, U, U), IntegrationError> {
        self.integrate_fixed_step(function, time, initial_condition)
    }
}

impl<Y, U> FixedStepExplicit<Y, U> for Ralston
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Add<Y, Output = Y>,
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
        *y_trial = &k[0] * (0.75 * dt) + y;
        k[1] = function(t + 0.75 * dt, y_trial)?;
        *y_trial = (&k[0] + &k[1] * 2.0) * (dt / 3.0) + y;
        Ok(())
    }
}
