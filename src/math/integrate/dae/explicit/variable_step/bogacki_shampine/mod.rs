use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::{BogackiShampine, ExplicitDaeVariableStep, ExplicitDaeVariableStepFirstSameAsLast},
};
use std::ops::{Mul, Sub};

impl<Y, Z, U, V> ExplicitDaeVariableStep<Y, Z, U, V> for BogackiShampine
where
    Self: ExplicitDaeVariableStepFirstSameAsLast<Y, Z, U, V>,
    Y: Tensor,
    Z: Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
{
    fn slopes_solve(
        mut evolution: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        mut solution: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        y: &Y,
        z: &Z,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
        z_trial: &mut Z,
    ) -> Result<(), String> {
        *y_trial = &k[0] * (0.5 * dt) + y;
        *z_trial = solution(t + 0.5 * dt, y_trial, z)?;
        k[1] = evolution(t + 0.5 * dt, y_trial, z_trial)?;
        *y_trial = &k[1] * (0.75 * dt) + y;
        *z_trial = solution(t + 0.75 * dt, y_trial, z_trial)?;
        k[2] = evolution(t + 0.75 * dt, y_trial, z_trial)?;
        *y_trial = (&k[0] * 2.0 + &k[1] * 3.0 + &k[2] * 4.0) * (dt / 9.0) + y;
        *z_trial = solution(t + dt, y_trial, z_trial)?;
        Ok(())
    }
    fn slopes_solve_and_error(
        &self,
        evolution: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        solution: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        y: &Y,
        z: &Z,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
        z_trial: &mut Z,
    ) -> Result<Scalar, String> {
        Self::slopes_solve_and_error_fsal(evolution, solution, y, z, t, dt, k, y_trial, z_trial)
    }
    fn step_solve(
        &self,
        _: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        y: &mut Y,
        z: &mut Z,
        t: &mut Scalar,
        y_sol: &mut U,
        z_sol: &mut V,
        t_sol: &mut Vector,
        dydt_sol: &mut U,
        dt: &mut Scalar,
        k: &mut [Y],
        y_trial: &Y,
        z_trial: &Z,
        e: Scalar,
    ) -> Result<(), String> {
        self.step_solve_fsal(
            y, z, t, y_sol, z_sol, t_sol, dydt_sol, dt, k, y_trial, z_trial, e,
        )
    }
}

impl<Y, Z, U, V> ExplicitDaeVariableStepFirstSameAsLast<Y, Z, U, V> for BogackiShampine
where
    Y: Tensor,
    Z: Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
{
}
