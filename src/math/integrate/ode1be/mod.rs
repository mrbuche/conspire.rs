#[cfg(test)]
mod test;

use super::{
    super::{
        optimize::{FirstOrder, NewtonRaphson, Optimization, SecondOrder},
        Hessian, Tensor, TensorArray, TensorRank0, TensorRank0List,
    },
    Implicit, IntegrationError, OdeSolver,
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
    /// Multiplying factor when increasing time steps.
    pub inc_fac: TensorRank0,
    /// Optimization algorithm for equation solving.
    pub optimization: Optimization,
    /// Relative error tolerance.
    pub rel_tol: TensorRank0,
}

impl Default for Ode1be {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            dec_fac: 0.5,
            inc_fac: 1.1,
            optimization: Optimization::NewtonRaphson(NewtonRaphson {
                check_minimum: false,
                ..Default::default()
            }),
            rel_tol: REL_TOL,
        }
    }
}

impl<Y, J, U, const W: usize> Implicit<Y, J, U, W> for Ode1be
where
    Y: Tensor + Div<J, Output = Y>,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    J: Hessian + Tensor + TensorArray,
    U: Tensor<Item = Y> + TensorArray,
{
    fn integrate(
        &self,
        function: impl Fn(&TensorRank0, &Y) -> Y,
        jacobian: impl Fn(&TensorRank0, &Y) -> J,
        initial_time: TensorRank0,
        initial_condition: Y,
        evaluation_times: &TensorRank0List<W>,
    ) -> Result<U, IntegrationError<W>> {
        let mut e;
        let mut k_1 = function(&initial_time, &initial_condition);
        let mut k_2;
        let mut solution = U::zero();
        let mut t_trial;
        let mut y_trial;
        let identity = J::identity();
        {
            let (mut eval_times, mut dt, mut t, mut y, mut y_sol) = self.setup(
                initial_time,
                initial_condition,
                evaluation_times,
                &mut solution,
            )?;
            while eval_times.peek().is_some() {
                t_trial = t + dt;
                y_trial = match &self.optimization {
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
                    while let Some(eval_time) = eval_times.next_if(|&eval_time| t > eval_time) {
                        *y_sol.next().ok_or("not ok")? =
                            (&y_trial - &y) / dt * (eval_time - t) + &y;
                    }
                    k_1 = k_2;
                    t += dt;
                    dt *= self.inc_fac;
                    y = y_trial;
                } else {
                    dt *= self.dec_fac;
                }
            }
        }
        Ok(solution)
    }
}
