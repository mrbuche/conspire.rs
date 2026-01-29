#[cfg(test)]
mod test;

use crate::math::{
    Scalar, Tensor, TensorArray, TensorVec,
    integrate::{FixedStep, ImplicitFirstOrder, ImplicitZerothOrder, IntegrationError, OdeSolver},
};
use std::{
    fmt::Debug,
    ops::{Mul, Sub},
};

#[doc = include_str!("doc.md")]
#[derive(Debug, Default)]
pub struct BackwardEuler {
    /// Fixed value for the time step.
    dt: Scalar,
}

impl<Y, U> OdeSolver<Y, U> for BackwardEuler
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl FixedStep for BackwardEuler {
    fn dt(&self) -> Scalar {
        self.dt
    }
}

impl<Y, U> ImplicitZerothOrder<Y, U> for BackwardEuler
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn residual(
        &self,
        mut function: impl FnMut(Scalar, &Y) -> Result<Y, IntegrationError>,
        _t: Scalar,
        y: &Y,
        t_trial: Scalar,
        y_trial: &Y,
        dt: Scalar,
    ) -> Result<Y, String> {
        Ok(y_trial - y - function(t_trial, y_trial)? * dt)
    }
}

impl<Y, J, U> ImplicitFirstOrder<Y, J, U> for BackwardEuler
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
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
        Ok(J::identity() - jacobian(t_trial, y_trial)? * dt)
    }
}
