use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::{Explicit, FixedStep, FixedStepExplicit, IntegrationError, OdeSolver},
};
use std::ops::{Add, Mul};

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

impl<Y, U> Explicit<Y, U> for Midpoint
where
    Self: OdeSolver<Y, U>,
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Add<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    const SLOPES: usize = 2;
    fn integrate(
        &self,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        time: &[Scalar],
        initial_condition: Y,
    ) -> Result<(Vector, U, U), IntegrationError> {
        self.integrate_fixed_step(function, time, initial_condition)
    }
}

impl<Y, U> FixedStepExplicit<Y, U> for Midpoint
where
    Self: OdeSolver<Y, U>,
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Add<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn step(
        &self,
        mut function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &mut Y,
        t: &mut Scalar,
        y_sol: &mut U,
        dydt_sol: &mut U,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
    ) -> Result<(), String> {
        k[0] = function(*t, y)?;
        *y_trial = &k[0] * (0.5 * dt) + y.clone();
        k[1] = function(*t + 0.5 * dt, y_trial)?;
        *y_trial = &k[1] * dt + y.clone();
        *t += dt;
        *y = y_trial.clone();
        y_sol.push(y.clone());
        dydt_sol.push(k[0].clone());
        Ok(())
    }
}
