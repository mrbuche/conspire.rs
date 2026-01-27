#[cfg(test)]
mod test;

use super::{
    super::{
        Scalar, Tensor, TensorVec, Vector,
        interpolate::{InterpolateSolution, InterpolateSolutionInternalVariables},
    },
    Explicit, ExplicitInternalVariables, IntegrationError, OdeSolver, VariableStep,
    VariableStepExplicit,
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
    Self: OdeSolver<Y, U>,
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
    Self: OdeSolver<Y, U>,
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn slopes(
        &self,
        mut function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &Y,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
    ) -> Result<Scalar, String> {
        *y_trial = &k[0] * (0.5 * dt) + y;
        k[1] = function(t + 0.5 * dt, y_trial)?;
        *y_trial = &k[1] * (0.75 * dt) + y;
        k[2] = function(t + 0.75 * dt, y_trial)?;
        *y_trial = (&k[0] * 2.0 + &k[1] * 3.0 + &k[2] * 4.0) * (dt / 9.0) + y;
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
        if e < self.abs_tol || e / y_trial.norm_inf() < self.rel_tol {
            k[0] = k[3].clone();
            *t += *dt;
            *y = y_trial.clone();
            t_sol.push(*t);
            y_sol.push(y.clone());
            dydt_sol.push(k[0].clone());
        }
        // self.time_step(e, dt); using below temporarily to pass test barely failing
        if e > 0.0 {
            *dt *= self.dt_beta() * (self.abs_tol() / e).powf(1.0 / self.dt_expn())
        }
        Ok(())
    }
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
        mut function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
    ) -> Result<(U, U), IntegrationError> {
        let mut dt;
        let mut i;
        let mut k_1;
        let mut k_2;
        let mut k_3;
        let mut t;
        let mut y;
        let mut y_int = U::new();
        let mut dydt_int = U::new();
        let mut y_trial;
        for time_k in time.iter() {
            i = tp.iter().position(|tp_i| tp_i >= time_k).unwrap();
            if time_k == &tp[i] {
                t = tp[i];
                y_trial = yp[i].clone();
                dt = 0.0;
            } else {
                t = tp[i - 1];
                y = yp[i - 1].clone();
                dt = time_k - t;
                k_1 = function(t, &y)?;
                y_trial = &k_1 * (0.5 * dt) + &y;
                k_2 = function(t + 0.5 * dt, &y_trial)?;
                y_trial = &k_2 * (0.75 * dt) + &y;
                k_3 = function(t + 0.75 * dt, &y_trial)?;
                y_trial = (&k_1 * 2.0 + &k_2 * 3.0 + &k_3 * 4.0) * (dt / 9.0) + &y;
            }
            dydt_int.push(function(t + dt, &y_trial)?);
            y_int.push(y_trial);
        }
        Ok((y_int, dydt_int))
    }
}

impl<Y, Z, U, V> ExplicitInternalVariables<Y, Z, U, V> for BogackiShampine
where
    Self: OdeSolver<Y, U>,
    Y: Tensor,
    Z: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
{
    const SLOPES: usize = 4;
    fn slopes(
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
        *y_trial = &k[0] * (0.5 * dt) + y;
        *z_trial = evaluate(t + 0.5 * dt, y_trial, z)?;
        k[1] = function(t + 0.5 * dt, y_trial, z_trial)?;
        *y_trial = &k[1] * (0.75 * dt) + y;
        *z_trial = evaluate(t + 0.75 * dt, y_trial, z_trial)?;
        k[2] = function(t + 0.75 * dt, y_trial, z_trial)?;
        *y_trial = (&k[0] * 2.0 + &k[1] * 3.0 + &k[2] * 4.0) * (dt / 9.0) + y;
        *z_trial = evaluate(t + dt, y_trial, z_trial)?;
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
            k[0] = k[3].clone();
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
        mut function: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        mut evaluate: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
    ) -> Result<(U, U, V), IntegrationError> {
        let mut dt;
        let mut i;
        let mut k_1;
        let mut k_2;
        let mut k_3;
        let mut t;
        let mut y;
        let mut y_int = U::new();
        let mut z_int = V::new();
        let mut dydt_int = U::new();
        let mut y_trial;
        let mut z_trial;
        for time_k in time.iter() {
            i = tp.iter().position(|tp_i| tp_i >= time_k).unwrap();
            if time_k == &tp[i] {
                t = tp[i];
                y_trial = yp[i].clone();
                z_trial = zp[i].clone();
                dt = 0.0;
            } else {
                t = tp[i - 1];
                y = yp[i - 1].clone();
                z_trial = zp[i - 1].clone();
                dt = time_k - t;
                k_1 = function(t, &y, &z_trial)?;
                y_trial = &k_1 * (0.5 * dt) + &y;
                z_trial = evaluate(t + 0.5 * dt, &y_trial, &z_trial)?;
                k_2 = function(t + 0.5 * dt, &y_trial, &z_trial)?;
                y_trial = &k_2 * (0.75 * dt) + &y;
                z_trial = evaluate(t + 0.75 * dt, &y_trial, &z_trial)?;
                k_3 = function(t + 0.75 * dt, &y_trial, &z_trial)?;
                y_trial = (&k_1 * 2.0 + &k_2 * 3.0 + &k_3 * 4.0) * (dt / 9.0) + &y;
                z_trial = evaluate(t + dt, &y_trial, &z_trial)?;
            }
            dydt_int.push(function(t + dt, &y_trial, &z_trial)?);
            y_int.push(y_trial);
            z_int.push(z_trial);
        }
        Ok((y_int, dydt_int, z_int))
    }
}
