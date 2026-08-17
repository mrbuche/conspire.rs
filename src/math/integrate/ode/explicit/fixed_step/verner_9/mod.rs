#[cfg(test)]
mod test;

use crate::math::{
    Derivative, Differentiate, Quantity, Scalar, Tensor, TensorVec,
    integrate::{
        Explicit, FixedStep, FixedStepExplicit, IntegrationError, OdeIntegrator, Times,
        VariableStepExplicit, Verner9 as Verner9VariableStep,
    },
};
use std::ops::{Mul, Sub};

#[doc = include_str!("doc.md")]
#[derive(Debug, Default)]
pub struct Verner9 {
    /// Fixed value for the time step.
    dt: Scalar,
}

impl<Y, U> OdeIntegrator<Y, U> for Verner9
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl<T> FixedStep<T> for Verner9 {
    fn dt(&self) -> Quantity<T> {
        Quantity::new(self.dt)
    }
}

impl<Y, U, V, T> Explicit<Y, U, V, T> for Verner9
where
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    const SLOPES: usize = 15;
    fn integrate(
        &self,
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        time: &[Quantity<T>],
        initial_condition: Y,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        self.integrate_fixed_step(function, time, initial_condition)
    }
}

impl<Y, U, V, T> FixedStepExplicit<Y, U, V, T> for Verner9
where
    Verner9VariableStep: VariableStepExplicit<Y, U, V, T>,
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    fn step(
        &self,
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        y: &Y,
        t: Quantity<T>,
        dt: Quantity<T>,
        k: &mut [Derivative<Y, T>],
        y_trial: &mut Y,
    ) -> Result<(), String> {
        Verner9VariableStep::slopes(function, y, t, dt, k, y_trial)
    }
}
