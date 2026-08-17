#[cfg(test)]
mod test;

use crate::math::{
    Derivative, Differentiate, Quantity, Scalar, Tensor, TensorArray, TensorVec,
    integrate::{
        FixedStep, ImplicitFirstOrder, ImplicitZerothOrder, IntegrationError, OdeIntegrator,
    },
};
use std::{
    fmt::Debug,
    ops::{Mul, Sub},
};

#[doc = include_str!("doc.md")]
#[derive(Debug, Default)]
pub struct Trapezoidal {
    /// Fixed value for the time step.
    dt: Scalar,
}

impl<Y, U> OdeIntegrator<Y, U> for Trapezoidal
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl<T> FixedStep<T> for Trapezoidal {
    fn dt(&self) -> Quantity<T> {
        Quantity::new(self.dt)
    }
}

impl<Y, U, V, T> ImplicitZerothOrder<Y, U, V, T> for Trapezoidal
where
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Y: Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    fn residual(
        &self,
        mut function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, IntegrationError>,
        t: Quantity<T>,
        y: &Y,
        t_trial: Quantity<T>,
        y_trial: &Y,
        dt: Quantity<T>,
    ) -> Result<Y, String> {
        Ok(y_trial - y - (function(t, y)? + function(t_trial, y_trial)?) * (0.5 * dt))
    }
}

impl<Y, J, U, V, T> ImplicitFirstOrder<Y, J, U, V, T> for Trapezoidal
where
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    J: Differentiate<T> + Tensor + TensorArray,
    Derivative<J, T>: Mul<Quantity<T>, Output = J>,
    for<'a> &'a Y: Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    fn hessian(
        &self,
        mut jacobian: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<J, T>, IntegrationError>,
        _t: Quantity<T>,
        _y: &Y,
        t_trial: Quantity<T>,
        y_trial: &Y,
        dt: Quantity<T>,
    ) -> Result<J, String> {
        Ok(J::identity() - jacobian(t_trial, y_trial)? * (0.5 * dt))
    }
}
