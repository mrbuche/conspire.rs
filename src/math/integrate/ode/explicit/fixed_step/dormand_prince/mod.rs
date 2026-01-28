#[cfg(test)]
mod test;

use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::{
        Explicit, FixedStep, FixedStepExplicit, IntegrationError, OdeSolver,
        ode::explicit::variable_step::dormand_prince::slopes,
    },
};
use std::ops::{Mul, Sub};

#[doc = include_str!("doc.md")]
#[derive(Debug, Default)]
pub struct DormandPrince {
    /// Fixed value for the time step.
    dt: Scalar,
}

impl<Y, U> OdeSolver<Y, U> for DormandPrince
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl FixedStep for DormandPrince {
    fn dt(&self) -> Scalar {
        self.dt
    }
}

impl<Y, U> Explicit<Y, U> for DormandPrince
where
    Self: OdeSolver<Y, U>,
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    const SLOPES: usize = 6;
    fn integrate(
        &self,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        time: &[Scalar],
        initial_condition: Y,
    ) -> Result<(Vector, U, U), IntegrationError> {
        self.integrate_fixed_step(function, time, initial_condition)
    }
}

impl<Y, U> FixedStepExplicit<Y, U> for DormandPrince
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
