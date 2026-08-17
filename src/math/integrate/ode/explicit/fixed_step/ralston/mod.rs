#[cfg(test)]
mod test;

use crate::math::{
    Derivative, Differentiate, Quantity, Scalar, Tensor, TensorVec,
    integrate::{Explicit, FixedStep, FixedStepExplicit, IntegrationError, OdeIntegrator, Times},
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

impl<T> FixedStep<T> for Ralston {
    fn dt(&self) -> Quantity<T> {
        Quantity::new(self.dt)
    }
}

impl<Y, U, V, T> Explicit<Y, U, V, T> for Ralston
where
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Derivative<Y, T>: Add<Derivative<Y, T>, Output = Derivative<Y, T>>
        + Mul<Scalar, Output = Derivative<Y, T>>
        + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    const SLOPES: usize = 2;
    fn integrate(
        &self,
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        time: &[Quantity<T>],
        initial_condition: Y,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        self.integrate_fixed_step(function, time, initial_condition)
    }
}

impl<Y, U, V, T> FixedStepExplicit<Y, U, V, T> for Ralston
where
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Derivative<Y, T>: Add<Derivative<Y, T>, Output = Derivative<Y, T>>
        + Mul<Scalar, Output = Derivative<Y, T>>
        + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    fn step(
        &self,
        mut function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        y: &Y,
        t: Quantity<T>,
        dt: Quantity<T>,
        k: &mut [Derivative<Y, T>],
        y_trial: &mut Y,
    ) -> Result<(), String> {
        k[0] = function(t, y)?;
        *y_trial = &k[0] * (0.75 * dt) + y;
        k[1] = function(t + 0.75 * dt, y_trial)?;
        *y_trial = (&k[0] + &k[1] * 2.0) * (dt / 3.0) + y;
        Ok(())
    }
}
