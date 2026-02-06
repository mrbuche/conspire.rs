#[cfg(test)]
mod test;

use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::{Explicit, IntegrationError, VariableStep},
    interpolate::InterpolateSolution,
};
use std::ops::{Mul, Sub};

pub mod bogacki_shampine;
pub mod dormand_prince;
pub mod verner_8;
pub mod verner_9;

/// Variable-step explicit ordinary differential equation solvers.
pub trait VariableStepExplicit<Y, U>
where
    Self: InterpolateSolution<Y, U> + Explicit<Y, U> + VariableStep,
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn integrate_variable_step(
        &self,
        mut function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        time: &[Scalar],
        initial_condition: Y,
    ) -> Result<(Vector, U, U), IntegrationError> {
        let t_0 = time[0];
        let t_f = time[time.len() - 1];
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        } else if t_0 >= t_f {
            return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
        }
        let mut t = t_0;
        let mut dt = t_f - t_0;
        let mut k = vec![Y::default(); Self::SLOPES];
        k[0] = function(t, &initial_condition)?;
        let mut t_sol = Vector::new();
        t_sol.push(t_0);
        let mut y = initial_condition.clone();
        let mut y_sol = U::new();
        y_sol.push(initial_condition.clone());
        let mut dydt_sol = U::new();
        dydt_sol.push(k[0].clone());
        let mut y_trial = Y::default();
        while t < t_f {
            match self.slopes_and_error(&mut function, &y, t, dt, &mut k, &mut y_trial) {
                Ok(e) => {
                    if let Err(error) = self.step(
                        &mut function,
                        &mut y,
                        &mut t,
                        &mut y_sol,
                        &mut t_sol,
                        &mut dydt_sol,
                        &mut dt,
                        &mut k,
                        &y_trial,
                        e,
                    ) {
                        dt *= self.dt_cut();
                        if dt < self.dt_min() {
                            return Err(IntegrationError::MinimumStepSizeUpstream(
                                self.dt_min(),
                                error,
                                format!("{self:?}"),
                            ));
                        }
                    } else {
                        dt = dt.min(t_f - t);
                        if dt < self.dt_min() && t < t_f {
                            return Err(IntegrationError::MinimumStepSizeReached(
                                self.dt_min(),
                                format!("{self:?}"),
                            ));
                        }
                    }
                }
                Err(error) => {
                    dt *= self.dt_cut();
                    if dt < self.dt_min() {
                        return Err(IntegrationError::MinimumStepSizeUpstream(
                            self.dt_min(),
                            error,
                            format!("{self:?}"),
                        ));
                    }
                }
            }
        }
        if time.len() > 2 {
            let t_int = Vector::from(time);
            let (y_int, dydt_int) = self.interpolate(&t_int, &t_sol, &y_sol, function)?;
            Ok((t_int, y_int, dydt_int))
        } else {
            Ok((t_sol, y_sol, dydt_sol))
        }
    }
    fn interpolate_variable_step(
        time: &Vector,
        tp: &Vector,
        yp: &U,
        mut function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
    ) -> Result<(U, U), IntegrationError> {
        let mut dt;
        let mut i;
        let mut k = vec![Y::default(); Self::SLOPES];
        let mut t;
        let mut y;
        let mut y_int = U::new();
        let mut dydt_int = U::new();
        let mut y_trial = Y::default();
        for time_k in time.iter() {
            i = tp.iter().position(|tp_i| tp_i >= time_k).unwrap();
            if time_k == &tp[i] {
                t = tp[i];
                y_trial = yp[i].clone();
                dt = 0.0;
            } else {
                t = tp[i - 1];
                y = &yp[i - 1];
                dt = time_k - t;
                k[0] = function(t, y)?;
                Self::slopes(&mut function, y, t, dt, &mut k, &mut y_trial)?;
            }
            dydt_int.push(function(t + dt, &y_trial)?);
            y_int.push(y_trial.clone());
        }
        Ok((y_int, dydt_int))
    }
    fn error(dt: Scalar, k: &[Y]) -> Result<Scalar, String>;
    fn slopes(
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &Y,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
    ) -> Result<(), String>;
    fn slopes_and_error(
        &self,
        mut function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &Y,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
    ) -> Result<Scalar, String> {
        Self::slopes(&mut function, y, t, dt, k, y_trial)?;
        Self::error(dt, k)
    }
    #[allow(clippy::too_many_arguments)]
    fn step(
        &self,
        mut function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &mut Y,
        t: &mut Scalar,
        y_sol: &mut U,
        t_sol: &mut Vector,
        dydt_sol: &mut U,
        dt: &mut Scalar,
        _k: &mut [Y],
        y_trial: &Y,
        e: Scalar,
    ) -> Result<(), String> {
        if e < self.abs_tol() || e / y_trial.norm_inf() < self.rel_tol() {
            *t += *dt;
            *y = y_trial.clone();
            t_sol.push(*t);
            y_sol.push(y.clone());
            dydt_sol.push(function(*t, y)?);
        }
        self.time_step(e, dt);
        Ok(())
    }
    /// Provides the adaptive time step as a function of the error.
    ///
    /// ```math
    /// h_{n+1} = \beta h \left(\frac{e_\mathrm{tol}}{e_{n+1}}\right)^{1/p}
    /// ```
    fn time_step(&self, error: Scalar, dt: &mut Scalar) {
        if error > 0.0 {
            *dt *= (self.dt_beta() * (self.abs_tol() / error).powf(1.0 / self.dt_expn()))
                .max(self.dt_cut())
        }
    }
}

/// First-same-as-last property for variable-step explicit ordinary differential equation solvers.
pub trait VariableStepExplicitFirstSameAsLast<Y, U>
where
    Self: VariableStepExplicit<Y, U>,
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn slopes_and_error_fsal(
        mut function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &Y,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
    ) -> Result<Scalar, String> {
        Self::slopes(&mut function, y, t, dt, k, y_trial)?;
        k[Self::SLOPES - 1] = function(t + dt, y_trial)?;
        Self::error(dt, k)
    }
    #[allow(clippy::too_many_arguments)]
    fn step_fsal(
        &self,
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
        if e < self.abs_tol() || e / y_trial.norm_inf() < self.rel_tol() {
            k[0] = k[Self::SLOPES - 1].clone();
            *t += *dt;
            *y = y_trial.clone();
            t_sol.push(*t);
            y_sol.push(y.clone());
            dydt_sol.push(k[0].clone());
        }
        self.time_step(e, dt);
        Ok(())
    }
}
