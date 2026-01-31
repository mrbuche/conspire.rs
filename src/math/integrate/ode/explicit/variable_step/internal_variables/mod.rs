use crate::math::{
    Scalar, Tensor, TensorVec, Vector, assert_eq_within_tols,
    integrate::{IntegrationError, VariableStepExplicit},
    interpolate::InterpolateSolutionInternalVariables,
};
use std::ops::{Mul, Sub};

/// Variable-step explicit ordinary differential equation solvers with internal variables.
pub trait VariableStepExplicitInternalVariables<Y, Z, U, V>
where
    Self: InterpolateSolutionInternalVariables<Y, Z, U, V> + VariableStepExplicit<Y, U>,
    Y: Tensor,
    Z: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
{
    fn integrate_and_evaluate_variable_step(
        &self,
        mut function: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        mut evaluate: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        time: &[Scalar],
        initial_condition: Y,
        initial_evaluation: Z,
    ) -> Result<(Vector, U, U, V), IntegrationError> {
        let t_0 = time[0];
        let t_f = time[time.len() - 1];
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        } else if t_0 >= t_f {
            return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
        }
        let mut t = t_0;
        let mut dt = t_f - t_0;
        let mut t_sol = Vector::new();
        t_sol.push(t_0);
        let mut y = initial_condition;
        let mut z = initial_evaluation;
        if assert_eq_within_tols(&evaluate(t, &y, &z)?, &z).is_err() {
            return Err(IntegrationError::InconsistentInitialConditions);
        }
        let mut k = vec![Y::default(); Self::SLOPES];
        k[0] = function(t, &y, &z)?;
        let mut y_sol = U::new();
        y_sol.push(y.clone());
        let mut z_sol = V::new();
        z_sol.push(z.clone());
        let mut dydt_sol = U::new();
        dydt_sol.push(k[0].clone());
        let mut y_trial = Y::default();
        let mut z_trial = Z::default();
        while t < t_f {
            match self.slopes_and_eval_and_error(
                &mut function,
                &mut evaluate,
                &y,
                &z,
                t,
                dt,
                &mut k,
                &mut y_trial,
                &mut z_trial,
            ) {
                Ok(e) => {
                    if let Some(error) = self
                        .step_and_eval(
                            &mut function,
                            &mut y,
                            &mut z,
                            &mut t,
                            &mut y_sol,
                            &mut z_sol,
                            &mut t_sol,
                            &mut dydt_sol,
                            &mut dt,
                            &mut k,
                            &y_trial,
                            &z_trial,
                            e,
                        )
                        .err()
                    {
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
            let (y_int, dydt_int, z_int) =
                self.interpolate_and_evaluate(&t_int, &t_sol, &y_sol, &z_sol, function, evaluate)?;
            Ok((t_int, y_int, dydt_int, z_int))
        } else {
            Ok((t_sol, y_sol, dydt_sol, z_sol))
        }
    }
    fn interpolate_and_evaluate_variable_step(
        time: &Vector,
        tp: &Vector,
        yp: &U,
        zp: &V,
        mut function: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        mut evaluate: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
    ) -> Result<(U, U, V), IntegrationError> {
        let mut dt;
        let mut i;
        let mut k = vec![Y::default(); Self::SLOPES];
        let mut t;
        let mut y;
        let mut z;
        let mut y_int = U::new();
        let mut z_int = V::new();
        let mut dydt_int = U::new();
        let mut y_trial = Y::default();
        let mut z_trial = Z::default();
        for time_k in time.iter() {
            i = tp.iter().position(|tp_i| tp_i >= time_k).unwrap();
            if time_k == &tp[i] {
                t = tp[i];
                y_trial = yp[i].clone();
                z_trial = zp[i].clone();
                dt = 0.0;
            } else {
                t = tp[i - 1];
                y = &yp[i - 1];
                z = &zp[i - 1];
                dt = time_k - t;
                k[0] = function(t, y, z)?;
                Self::slopes_and_eval(
                    &mut function,
                    &mut evaluate,
                    y,
                    z,
                    t,
                    dt,
                    &mut k,
                    &mut y_trial,
                    &mut z_trial,
                )?;
            }
            dydt_int.push(function(t + dt, &y_trial, &z_trial)?);
            y_int.push(y_trial.clone());
            z_int.push(z_trial.clone());
        }
        Ok((y_int, dydt_int, z_int))
    }
    #[allow(clippy::too_many_arguments)]
    fn slopes_and_eval(
        function: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        evaluate: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        y: &Y,
        z: &Z,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
        z_trial: &mut Z,
    ) -> Result<(), String>;
    #[allow(clippy::too_many_arguments)]
    fn slopes_and_eval_and_error(
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
        Self::error(dt, k)
    }
    #[allow(clippy::too_many_arguments)]
    fn step_and_eval(
        &self,
        mut function: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        y: &mut Y,
        z: &mut Z,
        t: &mut Scalar,
        y_sol: &mut U,
        z_sol: &mut V,
        t_sol: &mut Vector,
        dydt_sol: &mut U,
        dt: &mut Scalar,
        _k: &mut [Y],
        y_trial: &Y,
        z_trial: &Z,
        e: Scalar,
    ) -> Result<(), String> {
        if e < self.abs_tol() || e / y_trial.norm_inf() < self.rel_tol() {
            *t += *dt;
            *y = y_trial.clone();
            *z = z_trial.clone();
            t_sol.push(*t);
            y_sol.push(y.clone());
            z_sol.push(z.clone());
            dydt_sol.push(function(*t, y, z)?);
        }
        self.time_step(e, dt);
        Ok(())
    }
}

/// First-same-as-last property for variable-step explicit ordinary differential equation solvers with internal variables.
pub trait VariableStepExplicitInternalVariablesFirstSameAsLast<Y, Z, U, V>
where
    Self: VariableStepExplicitInternalVariables<Y, Z, U, V>,
    Y: Tensor,
    Z: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
{
    #[allow(clippy::too_many_arguments)]
    fn slopes_and_eval_and_error_fsal(
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
        k[Self::SLOPES - 1] = function(t + dt, y_trial, z_trial)?;
        Self::error(dt, k)
    }
    #[allow(clippy::too_many_arguments)]
    fn step_and_eval_fsal(
        &self,
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
        if e < self.abs_tol() || e / y_trial.norm_inf() < self.rel_tol() {
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
