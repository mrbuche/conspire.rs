use crate::math::{
    Scalar, Tensor, TensorVec, Vector, assert_eq_within_tols,
    integrate::{DaeSolver, IntegrationError, VariableStepExplicit},
    optimize::{EqualityConstraint, ZerothOrderRootFinding},
};
use std::ops::{Mul, Sub};

pub mod bogacki_shampine;
pub mod dormand_prince;
pub mod verner_8;
pub mod verner_9;

pub trait VariableStepExplicitDaeSolver<Y, Z, U, V>
where
    Self: DaeSolver<Y, Z, U, V> + VariableStepExplicit<Y, U>,
    Y: Tensor,
    Z: Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
{
    fn interpolate_dae_variable_step(
        &self,
        mut evolution: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        mut solution: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        time: &Vector,
        tp: &Vector,
        yp: &U,
        zp: &V,
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
                k[0] = evolution(t, y, z)?;
                Self::slopes_solve(
                    &mut evolution,
                    &mut solution,
                    y,
                    z,
                    t,
                    dt,
                    &mut k,
                    &mut y_trial,
                    &mut z_trial,
                )?;
            }
            dydt_int.push(evolution(t + dt, &y_trial, &z_trial)?);
            y_int.push(y_trial.clone());
            z_int.push(z_trial.clone());
        }
        Ok((y_int, dydt_int, z_int))
    }
    #[allow(clippy::too_many_arguments)]
    fn slopes_solve(
        evolution: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        solution: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        y: &Y,
        z: &Z,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
        z_trial: &mut Z,
    ) -> Result<(), String>;
    #[allow(clippy::too_many_arguments)]
    fn slopes_solve_and_error(
        &self,
        mut evolution: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        mut solution: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        y: &Y,
        z: &Z,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
        z_trial: &mut Z,
    ) -> Result<Scalar, String> {
        Self::slopes_solve(
            &mut evolution,
            &mut solution,
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
    fn step_solve(
        &self,
        mut evolution: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
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
            dydt_sol.push(evolution(*t, y, z)?);
        }
        self.time_step(e, dt);
        Ok(())
    }
}

pub trait VariableStepExplicitDaeSolverFirstSameAsLast<Y, Z, U, V>
where
    Self: VariableStepExplicitDaeSolver<Y, Z, U, V>,
    Y: Tensor,
    Z: Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
{
    #[allow(clippy::too_many_arguments)]
    fn slopes_solve_and_error_fsal(
        mut evolution: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        mut solution: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        y: &Y,
        z: &Z,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
        z_trial: &mut Z,
    ) -> Result<Scalar, String> {
        Self::slopes_solve(
            &mut evolution,
            &mut solution,
            y,
            z,
            t,
            dt,
            k,
            y_trial,
            z_trial,
        )?;
        k[Self::SLOPES - 1] = evolution(t + dt, y_trial, z_trial)?;
        Self::error(dt, k)
    }
    #[allow(clippy::too_many_arguments)]
    fn step_solve_fsal(
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

pub trait VariableStepExplicitDaeSolverZerothOrderRoot<Y, Z, U, V>
where
    Self: DaeSolver<Y, Z, U, V> + VariableStepExplicitDaeSolver<Y, Z, U, V>,
    Y: Tensor,
    Z: Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
{
    fn integrate_dae_variable_step(
        &self,
        mut evolution: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        mut function: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        solver: impl ZerothOrderRootFinding<Z>,
        time: &[Scalar],
        initial_condition: (Y, Z),
        mut equality_constraint: impl FnMut(Scalar) -> EqualityConstraint,
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
        let (mut y, mut z) = initial_condition;
        let mut solution = |t: Scalar, y: &Y, z_0: &Z| -> Result<Z, String> {
            Ok(solver.root(|z| function(t, y, z), z_0.clone(), equality_constraint(t))?)
        };
        if assert_eq_within_tols(&solution(t, &y, &z)?, &z).is_err() {
            return Err(IntegrationError::InconsistentInitialConditions);
        }
        let mut k = vec![Y::default(); Self::SLOPES];
        k[0] = evolution(t, &y, &z)?;
        let mut y_sol = U::new();
        y_sol.push(y.clone());
        let mut z_sol = V::new();
        z_sol.push(z.clone());
        let mut dydt_sol = U::new();
        dydt_sol.push(k[0].clone());
        let mut y_trial = Y::default();
        let mut z_trial = Z::default();
        while t < t_f {
            match self.slopes_solve_and_error(
                &mut evolution,
                &mut solution,
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
                        .step_solve(
                            &mut evolution,
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
            let (y_int, dydt_int, z_int) = self.interpolate_dae_variable_step(
                evolution, solution, &t_int, &t_sol, &y_sol, &z_sol,
            )?;
            Ok((t_int, y_int, dydt_int, z_int))
        } else {
            Ok((t_sol, y_sol, dydt_sol, z_sol))
        }
    }
}
