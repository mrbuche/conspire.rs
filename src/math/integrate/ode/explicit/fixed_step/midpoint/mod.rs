#[cfg(test)]
mod test;

use crate::math::{
    Derivative, Differentiate, Quantity, Scalar, Tensor, TensorVec,
    integrate::{
        ButcherTableau, Explicit, FixedStep, FixedStepExplicit, IntegrationError, OdeIntegrator,
        Times,
    },
};
use std::ops::Mul;

/// The explicit midpoint tableau.
#[derive(Debug)]
pub struct Tableau;

impl ButcherTableau for Tableau {
    const STAGES: usize = 2;
    const ORDER: Scalar = 2.0;
    const A: &'static [&'static [Scalar]] = &[&[], &[0.5]];
    const C: &'static [Scalar] = &[0.0, 0.5];
    const B: &'static [Scalar] = &[0.0, 1.0];
}

#[doc = include_str!("doc.md")]
#[derive(Debug, Default)]
pub struct Midpoint {
    /// Fixed value for the time step.
    dt: Scalar,
}

impl<Y, U> OdeIntegrator<Y, U> for Midpoint
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl<T> FixedStep<T> for Midpoint {
    fn dt(&self) -> Quantity<T> {
        Quantity::new(self.dt)
    }
}

impl<Y, U, V, T> Explicit<Y, U, V, T> for Midpoint
where
    Y: Differentiate<T> + Tensor,
    for<'a> &'a Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
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

impl<Y, U, V, T> FixedStepExplicit<Y, U, V, T> for Midpoint
where
    Y: Differentiate<T> + Tensor,
    for<'a> &'a Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    type Tableau = Tableau;
}
