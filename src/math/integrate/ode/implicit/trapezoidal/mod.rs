#[cfg(test)]
mod test;

use crate::math::{
    Scalar, Tensor, TensorArray, TensorVec,
    integrate::{
        FixedStep, ImplicitFirstOrder, ImplicitZerothOrder, IntegrationError, OdeIntegrator,
    },
};
use std::{
    fmt::Debug,
    ops::{Add, Mul, Sub},
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

impl FixedStep for Trapezoidal {
    fn dt(&self) -> Scalar {
        self.dt
    }
}

impl<Y, U> ImplicitZerothOrder<Y, U> for Trapezoidal
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Add<&'a Y, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn residual(
        &self,
        mut function: impl FnMut(Scalar, &Y) -> Result<Y, IntegrationError>,
        t: Scalar,
        y: &Y,
        t_trial: Scalar,
        y_trial: &Y,
        dt: Scalar,
    ) -> Result<Y, String> {
        Ok(y_trial - y - (function(t, y)? + function(t_trial, y_trial)?) * (0.5 * dt))
    }
}

impl<Y, J, U> ImplicitFirstOrder<Y, J, U> for Trapezoidal
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Add<&'a Y, Output = Y> + Sub<&'a Y, Output = Y>,
    J: Tensor + TensorArray,
    U: TensorVec<Item = Y>,
{
    fn hessian(
        &self,
        mut jacobian: impl FnMut(Scalar, &Y) -> Result<J, IntegrationError>,
        _t: Scalar,
        _y: &Y,
        t_trial: Scalar,
        y_trial: &Y,
        dt: Scalar,
    ) -> Result<J, String> {
        Ok(J::identity() - jacobian(t_trial, y_trial)? * (0.5 * dt))
    }
}
