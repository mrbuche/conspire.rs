#[cfg(test)]
mod test;

use crate::math::{
    Derivative, Differentiate, Quantity, Scalar, Tensor, TensorVec,
    integrate::{Explicit, IntegrationError, Times, VariableStep},
    interpolate::InterpolateSolution,
};
use crate::units::Time;
use std::ops::{Div, Mul, Sub};

pub(crate) mod bogacki_shampine;
pub(crate) mod dormand_prince;
pub(crate) mod verner_8;
pub(crate) mod verner_9;

/// Variable-step explicit integrators for ordinary differential equations.
pub trait VariableStepExplicit<Y, U, V, T = Time>
where
    Self: InterpolateSolution<Y, U, V, T> + Explicit<Y, U, V, T> + VariableStep<T>,
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    fn integrate_variable_step(
        &self,
        mut function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        time: &[Quantity<T>],
        initial_condition: Y,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        let t_0 = time[0];
        let t_f = time[time.len() - 1];
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        } else if t_0 >= t_f {
            return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
        }
        let mut t = t_0;
        let mut dt = t_f - t_0;
        let mut k = vec![Derivative::<Y, T>::default(); Self::SLOPES];
        k[0] = function(t, &initial_condition)?;
        let mut t_sol = Times::new();
        t_sol.push(t_0);
        let mut y = initial_condition.clone();
        let mut y_sol = U::new();
        y_sol.push(initial_condition.clone());
        let mut dydt_sol = V::new();
        dydt_sol.push(k[0].clone());
        let mut k_sol: Vec<V> = Vec::new();
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
                        &mut k_sol,
                        &mut dt,
                        &mut k,
                        &y_trial,
                        e,
                    ) {
                        dt *= self.dt_cut();
                        if dt < self.dt_min() {
                            return Err(IntegrationError::MinimumStepSizeUpstream(
                                self.dt_min().value(),
                                error,
                                format!("{self:?}"),
                            ));
                        }
                    } else {
                        dt = dt.min(t_f - t);
                        if dt < self.dt_min() && t < t_f {
                            return Err(IntegrationError::MinimumStepSizeReached(
                                self.dt_min().value(),
                                format!("{self:?}"),
                            ));
                        }
                    }
                }
                Err(error) => {
                    dt *= self.dt_cut();
                    if dt < self.dt_min() {
                        return Err(IntegrationError::MinimumStepSizeUpstream(
                            self.dt_min().value(),
                            error,
                            format!("{self:?}"),
                        ));
                    }
                }
            }
        }
        if time.len() > 2 {
            let t_int = Times::from(time);
            let (y_int, dydt_int) =
                self.interpolate(&t_int, &t_sol, &y_sol, &dydt_sol, &k_sol, function)?;
            Ok((t_int, y_int, dydt_int))
        } else {
            Ok((t_sol, y_sol, dydt_sol))
        }
    }
    fn interpolate_variable_step(
        time: &Times<T>,
        tp: &Times<T>,
        yp: &U,
        mut function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
    ) -> Result<(U, V), IntegrationError> {
        let mut dt;
        let mut i;
        let mut k = vec![Derivative::<Y, T>::default(); Self::SLOPES];
        let mut t;
        let mut y;
        let mut y_int = U::new();
        let mut dydt_int = V::new();
        let mut y_trial = Y::default();
        for time_k in time.iter() {
            i = tp.iter().position(|tp_i| tp_i >= time_k).unwrap();
            if time_k == &tp[i] {
                t = tp[i];
                y_trial = yp[i].clone();
                dt = Quantity::default();
            } else {
                t = tp[i - 1];
                y = &yp[i - 1];
                dt = *time_k - t;
                k[0] = function(t, y)?;
                Self::slopes(&mut function, y, t, dt, &mut k, &mut y_trial)?;
            }
            dydt_int.push(function(t + dt, &y_trial)?);
            y_int.push(y_trial.clone());
        }
        Ok((y_int, dydt_int))
    }
    fn error(&self, dt: Quantity<T>, k: &[Derivative<Y, T>]) -> Result<Scalar, String>;
    fn slopes(
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        y: &Y,
        t: Quantity<T>,
        dt: Quantity<T>,
        k: &mut [Derivative<Y, T>],
        y_trial: &mut Y,
    ) -> Result<(), String>;
    fn slopes_and_error(
        &self,
        mut function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        y: &Y,
        t: Quantity<T>,
        dt: Quantity<T>,
        k: &mut [Derivative<Y, T>],
        y_trial: &mut Y,
    ) -> Result<Scalar, String> {
        Self::slopes(&mut function, y, t, dt, k, y_trial)?;
        self.error(dt, k)
    }
    #[allow(clippy::too_many_arguments)]
    fn step(
        &self,
        mut function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        y: &mut Y,
        t: &mut Quantity<T>,
        y_sol: &mut U,
        t_sol: &mut Times<T>,
        dydt_sol: &mut V,
        k_sol: &mut Vec<V>,
        dt: &mut Quantity<T>,
        k: &mut [Derivative<Y, T>],
        y_trial: &Y,
        e: Scalar,
    ) -> Result<(), String> {
        if e < self.abs_tol() || e < self.rel_tol() * self.error_norm().measure(y_trial) {
            k_sol.push(k.iter().cloned().collect());
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
    fn time_step(&self, error: Scalar, dt: &mut Quantity<T>) {
        if error > 0.0 {
            *dt *= (self.dt_beta() * (self.abs_tol() / error).powf(1.0 / self.dt_expn()))
                .max(self.dt_cut())
        }
    }
}

/// Free (dense-output) interpolant for explicit ordinary differential equation integrators.
///
/// Uses cubic Hermite interpolation over the accepted-step values and derivatives already
/// computed during integration, so it requires no additional evaluations of the right-hand side
/// function.
pub trait FreeInterpolant<Y, U, V, T = Time>
where
    Self: VariableStepExplicit<Y, U, V, T>,
    Y: Differentiate<T> + Div<Quantity<T>, Output = Derivative<Y, T>> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    fn interpolate_free(time: &Times<T>, tp: &Times<T>, yp: &U, dydtp: &V) -> (U, V) {
        let mut y_int = U::new();
        let mut dydt_int = V::new();
        for time_k in time.iter() {
            let i = tp.iter().position(|tp_i| tp_i >= time_k).unwrap();
            if time_k == &tp[i] {
                y_int.push(yp[i].clone());
                dydt_int.push(dydtp[i].clone());
            } else {
                let t_0 = tp[i - 1];
                let h = tp[i] - t_0;
                let theta = (*time_k - t_0).value() / h.value();
                let theta2 = theta * theta;
                let theta3 = theta2 * theta;
                let h00 = 2.0 * theta3 - 3.0 * theta2 + 1.0;
                let h10 = theta3 - 2.0 * theta2 + theta;
                let h01 = -2.0 * theta3 + 3.0 * theta2;
                let h11 = theta3 - theta2;
                let dh00 = 6.0 * theta2 - 6.0 * theta;
                let dh10 = 3.0 * theta2 - 4.0 * theta + 1.0;
                let dh01 = -6.0 * theta2 + 6.0 * theta;
                let dh11 = 3.0 * theta2 - 2.0 * theta;
                y_int.push(
                    &yp[i - 1] * h00
                        + &dydtp[i - 1] * (h10 * h)
                        + &yp[i] * h01
                        + &dydtp[i] * (h11 * h),
                );
                dydt_int.push(
                    (&yp[i - 1] * dh00 + &yp[i] * dh01) / h
                        + &dydtp[i - 1] * dh10
                        + &dydtp[i] * dh11,
                );
            }
        }
        (y_int, dydt_int)
    }
}

/// First-same-as-last property for explicit ordinary differential equation integrators.
pub trait VariableStepExplicitFirstSameAsLast<Y, U, V, T = Time>
where
    Self: VariableStepExplicit<Y, U, V, T>,
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    fn slopes_and_error_fsal(
        &self,
        mut function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        y: &Y,
        t: Quantity<T>,
        dt: Quantity<T>,
        k: &mut [Derivative<Y, T>],
        y_trial: &mut Y,
    ) -> Result<Scalar, String> {
        Self::slopes(&mut function, y, t, dt, k, y_trial)?;
        k[Self::SLOPES - 1] = function(t + dt, y_trial)?;
        self.error(dt, k)
    }
    #[allow(clippy::too_many_arguments)]
    fn step_fsal(
        &self,
        y: &mut Y,
        t: &mut Quantity<T>,
        y_sol: &mut U,
        t_sol: &mut Times<T>,
        dydt_sol: &mut V,
        k_sol: &mut Vec<V>,
        dt: &mut Quantity<T>,
        k: &mut [Derivative<Y, T>],
        y_trial: &Y,
        e: Scalar,
    ) -> Result<(), String> {
        if e < self.abs_tol() || e < self.rel_tol() * self.error_norm().measure(y_trial) {
            k_sol.push(k.iter().cloned().collect());
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
