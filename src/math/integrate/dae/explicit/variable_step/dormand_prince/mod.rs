use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::{
        ExplicitDaeVariableStepExplicit, ExplicitDaeVariableStepFirstSameAsLast, IntegrationError,
        ode::explicit::variable_step::dormand_prince::*,
    },
};
use std::ops::{Mul, Sub};

impl<Y, Z, U, V> ExplicitDaeVariableStepExplicit<Y, Z, U, V> for DormandPrince
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
        *y_trial = &k[0] * (0.2 * dt) + y;
        *z_trial = solution(t + 0.2 * dt, y_trial, z)?;
        k[1] = evolution(t + 0.2 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (0.075 * dt) + &k[1] * (0.225 * dt) + y;
        *z_trial = solution(t + 0.3 * dt, y_trial, z_trial)?;
        k[2] = evolution(t + 0.3 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (C_44_45 * dt) - &k[1] * (C_56_15 * dt) + &k[2] * (C_32_9 * dt) + y;
        *z_trial = solution(t + 0.8 * dt, y_trial, z_trial)?;
        k[3] = evolution(t + 0.8 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (C_19372_6561 * dt) - &k[1] * (C_25360_2187 * dt)
            + &k[2] * (C_64448_6561 * dt)
            - &k[3] * (C_212_729 * dt)
            + y;
        *z_trial = solution(t + C_8_9 * dt, y_trial, z_trial)?;
        k[4] = evolution(t + C_8_9 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (C_9017_3168 * dt) - &k[1] * (C_355_33 * dt)
            + &k[2] * (C_46732_5247 * dt)
            + &k[3] * (C_49_176 * dt)
            - &k[4] * (C_5103_18656 * dt)
            + y;
        *z_trial = solution(t + dt, y_trial, z_trial)?;
        k[5] = evolution(t + dt, y_trial, z_trial)?;
        *y_trial = (&k[0] * C_35_384 + &k[2] * C_500_1113 + &k[3] * C_125_192
            - &k[4] * C_2187_6784
            + &k[5] * C_11_84)
            * dt
            + y;
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
        k_sol: &[U],
        zp: &V,
    ) -> Result<(U, U, V), IntegrationError> {
        let (y_int, dydt_int) = Self::interpolate_free_dense(time, tp, yp, dydtp, k_sol);
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

impl<Y, Z, U, V> ExplicitDaeVariableStepFirstSameAsLast<Y, Z, U, V> for DormandPrince
where
    Y: Tensor,
    Z: PartialEq + Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
{
}
