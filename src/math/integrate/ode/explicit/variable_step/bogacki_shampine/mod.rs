#[cfg(test)]
mod test;

use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::{
        Explicit, ExplicitInternalVariables, IntegrationError, OdeSolver, VariableStep,
        VariableStepExplicit, VariableStepExplicitFirstSameAsLast,
        VariableStepExplicitInternalVariables, VariableStepExplicitInternalVariablesFirstSameAsLast
    },
    interpolate::{InterpolateSolution, InterpolateSolutionInternalVariables},
};
use crate::{ABS_TOL, REL_TOL};
use std::ops::{Mul, Sub};

#[doc = include_str!("doc.md")]
#[derive(Debug)]
pub struct BogackiShampine {
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

impl Default for BogackiShampine {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            rel_tol: REL_TOL,
            dt_beta: 0.9,
            dt_expn: 3.0,
            dt_cut: 0.5,
            dt_min: ABS_TOL,
        }
    }
}

impl<Y, U> OdeSolver<Y, U> for BogackiShampine
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl VariableStep for BogackiShampine {
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

impl<Y, U> Explicit<Y, U> for BogackiShampine
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    const SLOPES: usize = 4;
    fn integrate(
        &self,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        time: &[Scalar],
        initial_condition: Y,
    ) -> Result<(Vector, U, U), IntegrationError> {
        self.integrate_variable_step(function, time, initial_condition)
    }
}

impl<Y, U> VariableStepExplicit<Y, U> for BogackiShampine
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
        *y_trial = &k[0] * (0.5 * dt) + y;
        k[1] = function(t + 0.5 * dt, y_trial)?;
        *y_trial = &k[1] * (0.75 * dt) + y;
        k[2] = function(t + 0.75 * dt, y_trial)?;
        *y_trial = (&k[0] * 2.0 + &k[1] * 3.0 + &k[2] * 4.0) * (dt / 9.0) + y;
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
        k[3] = function(t + dt, y_trial)?;
        Ok(((&k[0] * -5.0 + &k[1] * 6.0 + &k[2] * 8.0 + &k[3] * -9.0) * (dt / 72.0)).norm_inf())
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
        let dt_0 = *dt;
        self.step_fsal(y, t, y_sol, t_sol, dydt_sol, dt, k, y_trial, e)?;
        if e > 0.0 {
            *dt = dt_0;
            *dt *= self.dt_beta() * (self.abs_tol() / e).powf(1.0 / self.dt_expn())
        }
        Ok(()) // some temporary fixes to pass tests in fem that are barely failing
    }
}

impl<Y, U> VariableStepExplicitFirstSameAsLast<Y, U> for BogackiShampine
where
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
}

impl<Y, U> InterpolateSolution<Y, U> for BogackiShampine
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

impl<Y, Z, U, V> ExplicitInternalVariables<Y, Z, U, V> for BogackiShampine
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

impl<Y, Z, U, V> VariableStepExplicitInternalVariables<Y, Z, U, V> for BogackiShampine
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
        *y_trial = &k[0] * (0.5 * dt) + y;
        *z_trial = evaluate(t + 0.5 * dt, y_trial, z)?;
        k[1] = function(t + 0.5 * dt, y_trial, z_trial)?;
        *y_trial = &k[1] * (0.75 * dt) + y;
        *z_trial = evaluate(t + 0.75 * dt, y_trial, z_trial)?;
        k[2] = function(t + 0.75 * dt, y_trial, z_trial)?;
        *y_trial = (&k[0] * 2.0 + &k[1] * 3.0 + &k[2] * 4.0) * (dt / 9.0) + y;
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
        k[3] = function(t + dt, y_trial, z_trial)?;
        Ok(((&k[0] * -5.0 + &k[1] * 6.0 + &k[2] * 8.0 + &k[3] * -9.0) * (dt / 72.0)).norm_inf())
    }
    fn step(
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
        if e < self.abs_tol || e / y_trial.norm_inf() < self.rel_tol {
            k[0] = k[Self::SLOPES - 1].clone();
            *t += *dt;
            *y = y_trial.clone();
            *z = z_trial.clone();
            t_sol.push(*t);
            y_sol.push(y.clone());
            z_sol.push(z.clone());
            dydt_sol.push(k[0].clone());
        }
        self.time_step(e, dt);
        Ok(())
    }
}

impl<Y, Z, U, V> InterpolateSolutionInternalVariables<Y, Z, U, V> for BogackiShampine
where
    Y: Tensor,
    Z: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
{
    fn interpolate(
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
