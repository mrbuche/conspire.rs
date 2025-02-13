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
use std::ops::{Div, Mul, Sub};

/// Implicit, single-stage, first-order, fixed-step, Runge-Kutta method.[^cite]
///
/// [^cite]: Also known as the backward Euler method.
#[derive(Debug)]
pub struct BackwardEuler {
    /// Optimization algorithm for equation solving.
    pub opt_alg: Optimization,
}

impl Default for BackwardEuler {
    fn default() -> Self {
        Self {
            opt_alg: Optimization::NewtonRaphson(NewtonRaphson {
                check_minimum: false,
                ..Default::default()
            }),
        }
    }
}

impl<Y, J, U> Implicit<Y, J, U> for BackwardEuler
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
        time: &[TensorRank0],
        initial_condition: Y,
    ) -> Result<(Vector, U), IntegrationError> {
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        } else if time[0] >= time[time.len() - 1] {
            return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
        }
        let mut index = 0;
        let mut t = time[0];
        let mut dt;
        let identity = J::identity();
        let mut t_sol = Vector::zero(0);
        t_sol.push(time[0]);
        let mut t_trial;
        let mut y = initial_condition.copy();
        let mut y_sol = U::zero(0);
        y_sol.push(initial_condition.copy());
        let mut y_trial;
        while t < time[time.len() - 1] {
            t_trial = time[index + 1];
            dt = t_trial - t;
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
            t = t_trial;
            y = y_trial;
            t_sol.push(t.copy());
            y_sol.push(y.copy());
            index += 1;
        }
        Ok((t_sol, y_sol))
    }
}

impl<Y, U> InterpolateSolution<Y, U> for BackwardEuler
where
    Y: Tensor + TensorArray,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn interpolate(
        &self,
        _time: &Vector,
        _tp: &Vector,
        _yp: &U,
        _function: impl Fn(&TensorRank0, &Y) -> Y,
    ) -> U {
        unimplemented!()
    }
}
