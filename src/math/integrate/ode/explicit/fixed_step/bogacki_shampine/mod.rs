#[cfg(test)]
mod test;

use crate::math::{
    Derivative, Differentiate, Quantity, Scalar, Tensor, TensorVec,
    integrate::{
        BogackiShampine as BogackiShampineVariableStep, Explicit, FixedStep, FixedStepExplicit,
        IntegrationError, OdeIntegrator, Times, VariableStepExplicit,
    },
};
use std::ops::{Div, Mul, Sub};

#[doc = include_str!("doc.md")]
#[derive(Debug, Default)]
pub struct BogackiShampine {
    /// Fixed value for the time step.
    dt: Scalar,
}

impl<Y, U> OdeIntegrator<Y, U> for BogackiShampine
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl<T> FixedStep<T> for BogackiShampine {
    fn dt(&self) -> Quantity<T> {
        Quantity::new(self.dt)
    }
}

impl<Y, U, V, T> Explicit<Y, U, V, T> for BogackiShampine
where
    Y: Differentiate<T> + Div<Quantity<T>, Output = Derivative<Y, T>> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    const SLOPES: usize = 3;
    fn integrate(
        &self,
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        time: &[Quantity<T>],
        initial_condition: Y,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        self.integrate_fixed_step(function, time, initial_condition)
    }
}

impl<Y, U, V, T> FixedStepExplicit<Y, U, V, T> for BogackiShampine
where
    BogackiShampineVariableStep: VariableStepExplicit<Y, U, V, T>,
    Y: Differentiate<T> + Div<Quantity<T>, Output = Derivative<Y, T>> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
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
        BogackiShampineVariableStep::slopes(function, y, t, dt, k, y_trial)
    }
}
