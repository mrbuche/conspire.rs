#[cfg(test)]
mod test;

use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::{
        Explicit, FixedStep, FixedStepExplicit, IntegrationError, OdeSolver,
        ode::explicit::variable_step::bogacki_shampine::slopes,
    },
};
use std::ops::{Mul, Sub};

#[doc = include_str!("doc.md")]
#[derive(Debug, Default)]
pub struct BogackiShampine {
    /// Fixed value for the time step.
    dt: Scalar,
}

impl<Y, U> OdeSolver<Y, U> for BogackiShampine
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl FixedStep for BogackiShampine {
    fn dt(&self) -> Scalar {
        self.dt
    }
}

impl<Y, U> Explicit<Y, U> for BogackiShampine
where
    Self: OdeSolver<Y, U>,
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    const SLOPES: usize = 3;
    fn integrate(
        &self,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        time: &[Scalar],
        initial_condition: Y,
    ) -> Result<(Vector, U, U), IntegrationError> {
        self.integrate_fixed_step(function, time, initial_condition)
    }
}

impl<Y, U> FixedStepExplicit<Y, U> for BogackiShampine
where
    Self: OdeSolver<Y, U>,
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn step(
        &self,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &Y,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
    ) -> Result<(), String> {
        slopes(function, y, t, dt, k, y_trial)
    }
}
