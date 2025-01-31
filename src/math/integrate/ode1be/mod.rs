#[cfg(test)]
mod test;

use super::{
    super::{
        interpolate::InterpolateSolution,
        optimize::{FirstOrder, NewtonRaphson, Optimization, SecondOrder},
        Hessian, Tensor, TensorArray, TensorRank0, TensorVec, Vector,
    },
    Implicit, IntegrationError,
};
use crate::{ABS_TOL, REL_TOL};
use std::ops::{Div, Mul, Sub};

/// Implicit, single-stage, first-order, variable-step, Runge-Kutta method.[^cite]
///
/// [^cite]: Also known as the backward Euler method.
#[derive(Debug)]
pub struct Ode1be {
    /// Absolute error tolerance.
    pub abs_tol: TensorRank0,
    /// Multiplying factor when decreasing time steps.
    pub dec_fac: TensorRank0,
    /// Initial relative timestep.
    pub dt_init: TensorRank0,
    /// Multiplying factor when increasing time steps.
    pub inc_fac: TensorRank0,
    /// Optimization algorithm for equation solving.
    pub opt_alg: Optimization,
    /// Relative error tolerance.
    pub rel_tol: TensorRank0,
}

impl Default for Ode1be {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            dec_fac: 0.5,
            dt_init: 0.1,
            inc_fac: 1.1,
            opt_alg: Optimization::NewtonRaphson(NewtonRaphson {
                check_minimum: false,
                ..Default::default()
            }),
            rel_tol: REL_TOL,
        }
    }
}

impl<Y, J, U> Implicit<Y, J, U> for Ode1be
where
    Self: InterpolateSolution<Y, U>,
    Y: Tensor + TensorArray + Div<J, Output = Y>,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    J: Hessian + Tensor + TensorArray,
    U: TensorVec<Item = Y>,
{
    fn integrate(
        &self,
        function: impl Fn(&TensorRank0, &Y) -> Y,
        jacobian: impl Fn(&TensorRank0, &Y) -> J,
        initial_condition: Y,
        time: &[TensorRank0],
    ) -> Result<(Vector, U), IntegrationError> {
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        } else if time[0] >= time[time.len() - 1] {
            return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
        }
        let mut t = time[0];
        let mut dt = self.dt_init * time[time.len() - 1];
        let mut e;
        let identity = J::identity();
        let mut k_1 = function(&t, &initial_condition);
        let mut k_2;
        let mut t_sol = Vector::zero(0);
        t_sol.push(time[0]);
        let mut t_trial;
        let mut y = initial_condition.copy();
        let mut y_sol = U::zero(0);
        y_sol.push(initial_condition.copy());
        let mut y_trial;
        while t < time[time.len() - 1] {
            t_trial = t + dt;
            y_trial = match &self.opt_alg {
                Optimization::GradientDescent(gradient_descent) => gradient_descent
                    .minimize(
                        |y_trial: &Y| Ok(y_trial - &y - &(&function(&t_trial, y_trial) * dt)),
                        y.copy(),
                        None,
                        None,
                    )
                    .unwrap(),
                Optimization::NewtonRaphson(newton_raphson) => newton_raphson
                    .minimize(
                        |y_trial: &Y| Ok(y_trial - &y - &(&function(&t_trial, y_trial) * dt)),
                        |y_trial: &Y| Ok(jacobian(&t_trial, y_trial) * -dt + &identity),
                        y.copy(),
                        None,
                        None,
                    )
                    .unwrap(),
            };
            k_2 = function(&t_trial, &y_trial);
            e = ((&k_2 - &k_1) * (dt / 2.0)).norm();
            if e < self.abs_tol || e / y_trial.norm() < self.rel_tol {
                k_1 = k_2;
                t += dt;
                dt *= self.inc_fac;
                y = y_trial;
                t_sol.push(t.copy());
                y_sol.push(y.copy());
            } else {
                dt *= self.dec_fac;
            }
        }
        if time.len() > 2 {
            let t_int = Vector::new(time);
            let y_int = self.interpolate(&t_int, &t_sol, &y_sol, function);
            Ok((t_int, y_int))
        } else {
            Ok((t_sol, y_sol))
        }
    }
}

impl<Y, U> InterpolateSolution<Y, U> for Ode1be
where
    Y: Tensor + TensorArray,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn interpolate(
        &self,
        ti: &Vector,
        tp: &Vector,
        yp: &U,
        f: impl Fn(&TensorRank0, &Y) -> Y,
    ) -> U {
        let mut dt = 0.0;
        let mut i = 0;
        let mut t = 0.0;
        let mut y = Y::zero();
        ti.iter()
            .map(|ti_k| {
                i = tp.iter().position(|tp_i| tp_i > ti_k).unwrap();
                t = tp[i].copy();
                y = yp[i].copy();
                dt = ti_k - t;
                f(&t, &y) * dt + &y
            })
            .collect()
    }
}
