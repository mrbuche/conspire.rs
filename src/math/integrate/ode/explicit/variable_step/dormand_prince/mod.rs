#[cfg(test)]
mod test;

use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::{
        Explicit, ExplicitInternalVariables, IntegrationError, OdeSolver, VariableStep,
        VariableStepExplicit, VariableStepExplicitFirstSameAsLast,
        VariableStepExplicitInternalVariables,
        VariableStepExplicitInternalVariablesFirstSameAsLast,
    },
    interpolate::{InterpolateSolution, InterpolateSolutionInternalVariables},
};
use crate::{ABS_TOL, REL_TOL};
use std::ops::{Mul, Sub};

const C_44_45: Scalar = 44.0 / 45.0;
const C_56_15: Scalar = 56.0 / 15.0;
const C_32_9: Scalar = 32.0 / 9.0;
const C_8_9: Scalar = 8.0 / 9.0;
const C_19372_6561: Scalar = 19372.0 / 6561.0;
const C_25360_2187: Scalar = 25360.0 / 2187.0;
const C_64448_6561: Scalar = 64448.0 / 6561.0;
const C_212_729: Scalar = 212.0 / 729.0;
const C_9017_3168: Scalar = 9017.0 / 3168.0;
const C_355_33: Scalar = 355.0 / 33.0;
const C_46732_5247: Scalar = 46732.0 / 5247.0;
const C_49_176: Scalar = 49.0 / 176.0;
const C_5103_18656: Scalar = 5103.0 / 18656.0;
const C_35_384: Scalar = 35.0 / 384.0;
const C_500_1113: Scalar = 500.0 / 1113.0;
const C_125_192: Scalar = 125.0 / 192.0;
const C_2187_6784: Scalar = 2187.0 / 6784.0;
const C_11_84: Scalar = 11.0 / 84.0;
const C_71_57600: Scalar = 71.0 / 57600.0;
const C_71_16695: Scalar = 71.0 / 16695.0;
const C_71_1920: Scalar = 71.0 / 1920.0;
const C_17253_339200: Scalar = 17253.0 / 339200.0;
const C_22_525: Scalar = 22.0 / 525.0;

#[doc = include_str!("doc.md")]
#[derive(Debug)]
pub struct DormandPrince {
    /// Absolute error tolerance.
    pub abs_tol: Scalar,
    /// Relative error tolerance.
    pub rel_tol: Scalar,
    /// Multiplier for adaptive time steps.
    pub dt_beta: Scalar,
    /// Exponent for adaptive time steps.
    pub dt_expn: Scalar,
    /// Cut back factor for the time step.
    pub dt_cut: Scalar,
    /// Minimum value for the time step.
    pub dt_min: Scalar,
}

impl Default for DormandPrince {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            rel_tol: REL_TOL,
            dt_beta: 0.9,
            dt_expn: 5.0,
            dt_cut: 0.5,
            dt_min: ABS_TOL,
        }
    }
}

impl<Y, U> OdeSolver<Y, U> for DormandPrince
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl VariableStep for DormandPrince {
    fn abs_tol(&self) -> Scalar {
        self.abs_tol
    }
    fn rel_tol(&self) -> Scalar {
        self.rel_tol
    }
    fn dt_beta(&self) -> Scalar {
        self.dt_beta
    }
    fn dt_expn(&self) -> Scalar {
        self.dt_expn
    }
    fn dt_cut(&self) -> Scalar {
        self.dt_cut
    }
    fn dt_min(&self) -> Scalar {
        self.dt_min
    }
}

impl<Y, U> Explicit<Y, U> for DormandPrince
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    const SLOPES: usize = 7;
    fn integrate(
        &self,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        time: &[Scalar],
        initial_condition: Y,
    ) -> Result<(Vector, U, U), IntegrationError> {
        self.integrate_variable_step(function, time, initial_condition)
    }
}

impl<Y, U> VariableStepExplicit<Y, U> for DormandPrince
where
    Self: Explicit<Y, U>,
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn slopes(
        mut function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &Y,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
    ) -> Result<(), String> {
        *y_trial = &k[0] * (0.2 * dt) + y;
        k[1] = function(t + 0.2 * dt, y_trial)?;
        *y_trial = &k[0] * (0.075 * dt) + &k[1] * (0.225 * dt) + y;
        k[2] = function(t + 0.3 * dt, y_trial)?;
        *y_trial = &k[0] * (C_44_45 * dt) - &k[1] * (C_56_15 * dt) + &k[2] * (C_32_9 * dt) + y;
        k[3] = function(t + 0.8 * dt, y_trial)?;
        *y_trial = &k[0] * (C_19372_6561 * dt) - &k[1] * (C_25360_2187 * dt)
            + &k[2] * (C_64448_6561 * dt)
            - &k[3] * (C_212_729 * dt)
            + y;
        k[4] = function(t + C_8_9 * dt, y_trial)?;
        *y_trial = &k[0] * (C_9017_3168 * dt) - &k[1] * (C_355_33 * dt)
            + &k[2] * (C_46732_5247 * dt)
            + &k[3] * (C_49_176 * dt)
            - &k[4] * (C_5103_18656 * dt)
            + y;
        k[5] = function(t + dt, y_trial)?;
        *y_trial = (&k[0] * C_35_384 + &k[2] * C_500_1113 + &k[3] * C_125_192
            - &k[4] * C_2187_6784
            + &k[5] * C_11_84)
            * dt
            + y;
        Ok(())
    }
    fn slopes_with_error(
        &self,
        mut function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &Y,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
    ) -> Result<Scalar, String> {
        Self::slopes(&mut function, y, t, dt, k, y_trial)?;
        k[6] = function(t + dt, y_trial)?;
        Ok(
            ((&k[0] * C_71_57600 - &k[2] * C_71_16695 + &k[3] * C_71_1920
                - &k[4] * C_17253_339200
                + &k[5] * C_22_525
                - &k[6] * 0.025)
                * dt)
                .norm_inf(),
        )
    }
    fn step(
        &self,
        _function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &mut Y,
        t: &mut Scalar,
        y_sol: &mut U,
        t_sol: &mut Vector,
        dydt_sol: &mut U,
        dt: &mut Scalar,
        k: &mut [Y],
        y_trial: &Y,
        e: Scalar,
    ) -> Result<(), String> {
        self.step_fsal(y, t, y_sol, t_sol, dydt_sol, dt, k, y_trial, e)
    }
}

impl<Y, U> VariableStepExplicitFirstSameAsLast<Y, U> for DormandPrince
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
}

impl<Y, U> InterpolateSolution<Y, U> for DormandPrince
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn interpolate(
        &self,
        time: &Vector,
        tp: &Vector,
        yp: &U,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
    ) -> Result<(U, U), IntegrationError> {
        Self::interpolate_variable_step(time, tp, yp, function)
    }
}

impl<Y, Z, U, V> ExplicitInternalVariables<Y, Z, U, V> for DormandPrince
where
    Y: Tensor,
    Z: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
{
    fn integrate_and_evaluate(
        &self,
        function: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        evaluate: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        time: &[Scalar],
        initial_condition: Y,
        initial_evaluation: Z,
    ) -> Result<(Vector, U, U, V), IntegrationError> {
        self.integrate_and_evaluate_variable_step(
            function,
            evaluate,
            time,
            initial_condition,
            initial_evaluation,
        )
    }
}

impl<Y, Z, U, V> VariableStepExplicitInternalVariables<Y, Z, U, V> for DormandPrince
where
    Self: ExplicitInternalVariables<Y, Z, U, V>,
    Y: Tensor,
    Z: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
{
    fn slopes_and_eval(
        mut function: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        mut evaluate: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        y: &Y,
        z: &Z,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
        z_trial: &mut Z,
    ) -> Result<(), String> {
        *y_trial = &k[0] * (0.2 * dt) + y;
        *z_trial = evaluate(t + 0.2 * dt, y_trial, z)?;
        k[1] = function(t + 0.2 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (0.075 * dt) + &k[1] * (0.225 * dt) + y;
        *z_trial = evaluate(t + 0.3 * dt, y_trial, z_trial)?;
        k[2] = function(t + 0.3 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (C_44_45 * dt) - &k[1] * (C_56_15 * dt) + &k[2] * (C_32_9 * dt) + y;
        *z_trial = evaluate(t + 0.8 * dt, y_trial, z_trial)?;
        k[3] = function(t + 0.8 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (C_19372_6561 * dt) - &k[1] * (C_25360_2187 * dt)
            + &k[2] * (C_64448_6561 * dt)
            - &k[3] * (C_212_729 * dt)
            + y;
        *z_trial = evaluate(t + C_8_9 * dt, y_trial, z_trial)?;
        k[4] = function(t + C_8_9 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (C_9017_3168 * dt) - &k[1] * (C_355_33 * dt)
            + &k[2] * (C_46732_5247 * dt)
            + &k[3] * (C_49_176 * dt)
            - &k[4] * (C_5103_18656 * dt)
            + y;
        *z_trial = evaluate(t + dt, y_trial, z_trial)?;
        k[5] = function(t + dt, y_trial, z_trial)?;
        *y_trial = (&k[0] * C_35_384 + &k[2] * C_500_1113 + &k[3] * C_125_192
            - &k[4] * C_2187_6784
            + &k[5] * C_11_84)
            * dt
            + y;
        *z_trial = evaluate(t + dt, y_trial, z_trial)?;
        Ok(())
    }
    fn slopes_and_eval_with_error(
        &self,
        mut function: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        mut evaluate: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        y: &Y,
        z: &Z,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
        z_trial: &mut Z,
    ) -> Result<Scalar, String> {
        Self::slopes_and_eval(
            &mut function,
            &mut evaluate,
            y,
            z,
            t,
            dt,
            k,
            y_trial,
            z_trial,
        )?;
        k[6] = function(t + dt, y_trial, z_trial)?;
        Ok(
            ((&k[0] * C_71_57600 - &k[2] * C_71_16695 + &k[3] * C_71_1920
                - &k[4] * C_17253_339200
                + &k[5] * C_22_525
                - &k[6] * 0.025)
                * dt)
                .norm_inf(),
        )
    }
    fn step_and_eval(
        &self,
        _function: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
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
        self.step_and_eval_fsal(
            y, z, t, y_sol, z_sol, t_sol, dydt_sol, dt, k, y_trial, z_trial, e,
        )
    }
}

impl<Y, Z, U, V> VariableStepExplicitInternalVariablesFirstSameAsLast<Y, Z, U, V> for DormandPrince
where
    Y: Tensor,
    Z: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
{
}

impl<Y, Z, U, V> InterpolateSolutionInternalVariables<Y, Z, U, V> for DormandPrince
where
    Y: Tensor,
    Z: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
{
    fn interpolate_and_evaluate(
        &self,
        time: &Vector,
        tp: &Vector,
        yp: &U,
        zp: &V,
        function: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        evaluate: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
    ) -> Result<(U, U, V), IntegrationError> {
        Self::interpolate_and_evaluate_variable_step(time, tp, yp, zp, function, evaluate)
    }
}
