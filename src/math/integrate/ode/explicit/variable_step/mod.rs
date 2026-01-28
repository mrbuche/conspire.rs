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
            match self.slopes(&mut function, &y, t, dt, &mut k, &mut y_trial) {
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
    fn slopes(
        &self,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &Y,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
    ) -> Result<Scalar, String>;
    #[allow(clippy::too_many_arguments)]
    fn step(
        &self,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &mut Y,
        t: &mut Scalar,
        y_sol: &mut U,
        t_sol: &mut Vector,
        dydt_sol: &mut U,
        dt: &mut Scalar,
        k: &mut [Y],
        y_trial: &Y,
        e: Scalar,
    ) -> Result<(), String>;
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
