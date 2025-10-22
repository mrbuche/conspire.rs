#[cfg(test)]
mod test;

use super::{
    super::{
        Hessian, Jacobian, Scalar, Solution, TensorArray, TensorVec, Vector,
        interpolate::InterpolateSolution,
        optimize::{EqualityConstraint, FirstOrderRootFinding, ZerothOrderRootFinding},
    },
    ImplicitFirstOrder, ImplicitZerothOrder, IntegrationError,
};
use std::{
    fmt::Debug,
    ops::{Div, Mul, Sub},
};

/// Implicit, single-stage, first-order, fixed-step, Runge-Kutta method.[^cite]
///
/// [^cite]: Also known as the backward Euler method.
#[derive(Debug, Default)]
pub struct BackwardEuler {}

impl<Y, U> ImplicitZerothOrder<Y, U> for BackwardEuler
where
    Self: InterpolateSolution<Y, U>,
    Y: Solution,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
    Vector: From<Y>,
{
    fn integrate(
        &self,
        function: impl Fn(Scalar, &Y) -> Result<Y, IntegrationError>,
        time: &[Scalar],
        initial_condition: Y,
        solver: impl ZerothOrderRootFinding<Y>,
    ) -> Result<(Vector, U, U), IntegrationError> {
        let t_0 = time[0];
        let t_f = time[time.len() - 1];
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        } else if t_0 >= t_f {
            return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
        }
        let mut index = 0;
        let mut t = t_0;
        let mut dt;
        let mut t_sol = Vector::new();
        t_sol.push(t_0);
        let mut t_trial;
        let mut y = initial_condition.clone();
        let mut y_sol = U::new();
        y_sol.push(initial_condition.clone());
        let mut dydt_sol = U::new();
        dydt_sol.push(function(t, &y.clone())?);
        let mut y_trial;
        while t < t_f {
            t_trial = time[index + 1];
            dt = t_trial - t;
            y_trial = match solver.root(
                |y_trial: &Y| Ok(y_trial - &y - &(&function(t_trial, y_trial)? * dt)),
                y.clone(),
                EqualityConstraint::None,
            ) {
                Ok(solution) => solution,
                Err(error) => {
                    return Err(IntegrationError::Upstream(
                        format!("{error:?}"),
                        format!("{self:?}"),
                    ));
                }
            };
            t = t_trial;
            y = y_trial;
            t_sol.push(t);
            y_sol.push(y.clone());
            dydt_sol.push(function(t, &y)?);
            index += 1;
        }
        Ok((t_sol, y_sol, dydt_sol))
    }
}

impl<Y, J, U> ImplicitFirstOrder<Y, J, U> for BackwardEuler
where
    Self: InterpolateSolution<Y, U>,
    Y: Jacobian + Solution + Div<J, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    J: Hessian + TensorArray,
    U: TensorVec<Item = Y>,
    Vector: From<Y>,
{
    fn integrate(
        &self,
        function: impl Fn(Scalar, &Y) -> Result<Y, IntegrationError>,
        jacobian: impl Fn(Scalar, &Y) -> Result<J, IntegrationError>,
        time: &[Scalar],
        initial_condition: Y,
        solver: impl FirstOrderRootFinding<Y, J, Y>,
    ) -> Result<(Vector, U, U), IntegrationError> {
        let t_0 = time[0];
        let t_f = time[time.len() - 1];
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        } else if t_0 >= t_f {
            return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
        }
        let mut index = 0;
        let mut t = t_0;
        let mut dt;
        let identity = J::identity();
        let mut t_sol = Vector::new();
        t_sol.push(t_0);
        let mut t_trial;
        let mut y = initial_condition.clone();
        let mut y_sol = U::new();
        y_sol.push(initial_condition.clone());
        let mut dydt_sol = U::new();
        dydt_sol.push(function(t, &y.clone())?);
        let mut y_trial;
        while t < t_f {
            t_trial = time[index + 1];
            dt = t_trial - t;
            y_trial = match solver.root(
                |y_trial: &Y| Ok(y_trial - &y - &(&function(t_trial, y_trial)? * dt)),
                |y_trial: &Y| Ok(jacobian(t_trial, y_trial)? * -dt + &identity),
                y.clone(),
                EqualityConstraint::None,
            ) {
                Ok(solution) => solution,
                Err(error) => {
                    return Err(IntegrationError::Upstream(
                        format!("{error:?}"),
                        format!("{self:?}"),
                    ));
                }
            };
            t = t_trial;
            y = y_trial;
            t_sol.push(t);
            y_sol.push(y.clone());
            dydt_sol.push(function(t, &y)?);
            index += 1;
        }
        Ok((t_sol, y_sol, dydt_sol))
    }
}

impl<Y, U> InterpolateSolution<Y, U> for BackwardEuler
where
    Y: Jacobian + Solution + TensorArray,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
    Vector: From<Y>,
{
    fn interpolate(
        &self,
        _time: &Vector,
        _tp: &Vector,
        _yp: &U,
        mut _function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
    ) -> Result<(U, U), IntegrationError> {
        unimplemented!()
    }
}
