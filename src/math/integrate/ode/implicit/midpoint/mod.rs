#[cfg(test)]
mod test;

use crate::math::{
    Scalar, Tensor, TensorArray, TensorVec,
    integrate::{FixedStep, ImplicitFirstOrder, ImplicitZerothOrder, IntegrationError, OdeSolver},
};
use std::{
    fmt::Debug,
    ops::{Add, Mul, Sub},
};

#[doc = include_str!("doc.md")]
#[derive(Debug, Default)]
pub struct Midpoint {
    /// Fixed value for the time step.
    dt: Scalar,
}

impl<Y, U> OdeSolver<Y, U> for Midpoint
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl FixedStep for Midpoint {
    fn dt(&self) -> Scalar {
        self.dt
    }
}

impl<Y, U> ImplicitZerothOrder<Y, U> for Midpoint
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Add<&'a Y, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn residual(
        &self,
        function: impl Fn(Scalar, &Y) -> Result<Y, IntegrationError>,
        t: Scalar,
        y: &Y,
        _t_trial: Scalar,
        y_trial: &Y,
        dt: Scalar,
    ) -> Result<Y, String> {
        Ok(y_trial - y - function(t + 0.5 * dt, &((y + y_trial) * 0.5))? * dt)
    }
}

impl<Y, J, U> ImplicitFirstOrder<Y, J, U> for Midpoint
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Add<&'a Y, Output = Y> + Sub<&'a Y, Output = Y>,
    J: Tensor + TensorArray,
    U: TensorVec<Item = Y>,
{
    fn hessian(
        &self,
        jacobian: impl Fn(Scalar, &Y) -> Result<J, IntegrationError>,
        t: Scalar,
        y: &Y,
        _t_trial: Scalar,
        y_trial: &Y,
        dt: Scalar,
    ) -> Result<J, String> {
        Ok(J::identity() - jacobian(t + 0.5 * dt, &((y + y_trial) * 0.5))? * (dt * 0.5))
    }
}
