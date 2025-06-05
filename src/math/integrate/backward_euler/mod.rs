#[cfg(test)]
mod test;

use super::{
    super::{
        Hessian, Jacobian, Matrix, Solution, Tensor, TensorArray, TensorRank0, TensorVec, Vector,
        interpolate::InterpolateSolution,
        optimize::{
            EqualityConstraint, FirstOrderRootFinding, NewtonRaphson,
            ZerothOrderRootFinding,
        },
    },
    Implicit, IntegrationError,
};
use std::{fmt::Debug, ops::{Div, Mul, Sub}};

/// Implicit, single-stage, first-order, fixed-step, Runge-Kutta method.[^cite]
///
/// [^cite]: Also known as the backward Euler method.
#[derive(Debug)]
pub struct BackwardEuler<S> where S: Debug {
    /// Algorithm for equation solving.
    pub solver: S,
}

impl Default for BackwardEuler<NewtonRaphson> {
    fn default() -> Self {
        Self {
            solver: NewtonRaphson::default(),
        }
    }
}

macro_rules! implement_implicit {
    ($solver:ident) => {
        impl<Y, J, U> Implicit<Y, J, U> for BackwardEuler<$solver>
        where
            Self: InterpolateSolution<Y, U>,
            Y: Jacobian + Solution + Div<J, Output = Y>,
            for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
            J: Hessian + Tensor + TensorArray,
            U: TensorVec<Item = Y>,
            Vector: From<Y>,
            for<'a> &'a Matrix: Mul<&'a Y, Output = Vector>, // temporary until A replaced by sparse representation and similar implementation
        {
            fn integrate(
                &self,
                function: impl Fn(TensorRank0, &Y) -> Result<Y, IntegrationError>,
                jacobian: impl Fn(TensorRank0, &Y) -> Result<J, IntegrationError>,
                time: &[TensorRank0],
                initial_condition: Y,
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
                let mut t_sol = Vector::zero(0);
                t_sol.push(t_0);
                let mut t_trial;
                let mut y = initial_condition.clone();
                let mut y_sol = U::zero(0);
                y_sol.push(initial_condition.clone());
                let mut dydt_sol = U::zero(0);
                dydt_sol.push(function(t, &y.clone())?);
                let mut y_trial;
                while t < t_f {
                    t_trial = time[index + 1];
                    dt = t_trial - t;
                    y_trial = self.solver.root(
                        |y_trial: &Y| Ok(y_trial - &y - &(&function(t_trial, y_trial)? * dt)),
                        |y_trial: &Y| Ok(jacobian(t_trial, y_trial)? * -dt + &identity),
                        y.clone(),
                        EqualityConstraint::None,
                    )?;
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
    };
}

implement_implicit!(NewtonRaphson);

// y_trial = match &self.solver {
//     Solver::GradientDescent(gradient_descent) => gradient_descent.root(
//         |y_trial: &Y| Ok(y_trial - &y - &(&function(t_trial, y_trial)? * dt)),
//         y.clone(),
//         EqualityConstraint::None,
//     )?,
//     Solver::NewtonRaphson(newton_raphson) => newton_raphson.root(
//         |y_trial: &Y| Ok(y_trial - &y - &(&function(t_trial, y_trial)? * dt)),
//         |y_trial: &Y| Ok(jacobian(t_trial, y_trial)? * -dt + &identity),
//         y.clone(),
//         EqualityConstraint::None,
//     )?,
// };

impl<Y, U> InterpolateSolution<Y, U> for BackwardEuler<NewtonRaphson>
where
    Y: Jacobian + Solution + TensorArray,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
    Vector: From<Y>,
{
    fn interpolate(
        &self,
        _time: &Vector,
        _tp: &Vector,
        _yp: &U,
        _function: impl Fn(TensorRank0, &Y) -> Result<Y, IntegrationError>,
    ) -> Result<(U, U), IntegrationError> {
        unimplemented!()
    }
}
