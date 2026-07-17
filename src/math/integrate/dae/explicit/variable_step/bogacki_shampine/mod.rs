use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::{
        BogackiShampine, ExplicitDaeVariableStepExplicit, ExplicitDaeVariableStepFirstSameAsLast,
        FreeInterpolant, IntegrationError,
    },
};
use std::ops::{Mul, Sub};

impl<Y, Z, U, V> ExplicitDaeVariableStepExplicit<Y, Z, U, V> for BogackiShampine
where
    Self: ExplicitDaeVariableStepFirstSameAsLast<Y, Z, U, V>,
    Y: Tensor,
    Z: PartialEq + Tensor,
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
        self.slopes_solve_and_error_fsal(evolution, solution, y, z, t, dt, k, y_trial, z_trial)
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
        k_sol: &mut Vec<U>,
        dt: &mut Scalar,
        k: &mut [Y],
        y_trial: &Y,
        z_trial: &Z,
        e: Scalar,
    ) -> Result<(), String> {
        self.step_solve_fsal(
            y, z, t, y_sol, z_sol, t_sol, dydt_sol, k_sol, dt, k, y_trial, z_trial, e,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn interpolate_explicit_dae_variable_step(
        &self,
        _evolution: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        mut solution: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        time: &Vector,
        tp: &Vector,
        yp: &U,
        dydtp: &U,
        _k_sol: &[U],
        zp: &V,
    ) -> Result<(U, U, V), IntegrationError> {
        let (y_int, dydt_int) = Self::interpolate_free(time, tp, yp, dydtp);
        let mut z_int = V::new();
        for (idx, time_k) in time.iter().enumerate() {
            let i = tp.iter().position(|tp_i| tp_i >= time_k).unwrap();
            if time_k == &tp[i] {
                z_int.push(zp[i].clone());
            } else {
                z_int.push(solution(*time_k, &y_int[idx], &zp[i - 1])?);
            }
        }
        Ok((y_int, dydt_int, z_int))
    }
}

impl<Y, Z, U, V> ExplicitDaeVariableStepFirstSameAsLast<Y, Z, U, V> for BogackiShampine
where
    Y: Tensor,
    Z: PartialEq + Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
{
}
