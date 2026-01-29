#[cfg(test)]
mod test;

use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::{FixedStep, IntegrationError, OdeSolver},
    optimize::{EqualityConstraint, FirstOrderRootFinding, ZerothOrderRootFinding},
};

pub mod backward_euler;
pub mod midpoint;
pub mod trapezoidal;

/// Zeroth-order implicit ordinary differential equation solvers.
pub trait ImplicitZerothOrder<Y, U>
where
    Self: FixedStep + OdeSolver<Y, U>,
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
    #[doc = include_str!("doc.md")]
    fn integrate(
        &self,
        function: impl Fn(Scalar, &Y) -> Result<Y, IntegrationError>,
        time: &[Scalar],
        initial_condition: Y,
        solver: impl ZerothOrderRootFinding<Y>,
    ) -> Result<(Vector, U, U), IntegrationError> {
        let t_0 = time[0];
        let t_f = time[time.len() - 1];
        let mut t_sol: Vector;
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        } else if t_0 >= t_f {
            return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
        } else if time.len() == 2 {
            if self.dt() <= 0.0 || self.dt().is_nan() {
                return Err(IntegrationError::TimeStepNotSet(
                    time[0],
                    time[1],
                    format!("{self:?}"),
                ));
            } else {
                let num_steps = ((t_f - t_0) / self.dt()).ceil() as usize;
                t_sol = (0..num_steps)
                    .map(|step| t_0 + (step as Scalar) * self.dt())
                    .collect();
                t_sol.push(t_f);
            }
        } else {
            t_sol = time.iter().copied().collect();
        }
        let mut index = 0;
        let mut t = t_0;
        let mut dt;
        let mut t_trial;
        let mut y = initial_condition.clone();
        let mut y_sol = U::new();
        y_sol.push(initial_condition.clone());
        let mut dydt_sol = U::new();
        dydt_sol.push(function(t, &y.clone())?);
        let mut y_trial;
        while t < t_f {
            t_trial = t_sol[index + 1];
            dt = t_trial - t;
            y_trial = match solver.root(
                |y_trial: &Y| self.residual(&function, t, &y, t_trial, y_trial, dt),
                y.clone(),
                EqualityConstraint::None,
            ) {
                Ok(solution) => solution,
                Err(error) => {
                    return Err(IntegrationError::Upstream(
                        format!("{error}"),
                        format!("{self:?}"),
                    ));
                }
            };
            t = t_trial;
            y = y_trial;
            y_sol.push(y.clone());
            dydt_sol.push(function(t, &y)?);
            index += 1;
        }
        Ok((t_sol, y_sol, dydt_sol))
    }
    fn residual(
        &self,
        function: impl Fn(Scalar, &Y) -> Result<Y, IntegrationError>,
        t: Scalar,
        y: &Y,
        t_trial: Scalar,
        y_trial: &Y,
        dt: Scalar,
    ) -> Result<Y, String>;
}

/// First-order implicit ordinary differential equation solvers.
pub trait ImplicitFirstOrder<Y, J, U>
where
    Self: ImplicitZerothOrder<Y, U>,
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
    #[doc = include_str!("doc.md")]
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
        let mut t_sol: Vector;
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        } else if t_0 >= t_f {
            return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
        } else if time.len() == 2 {
            if self.dt() <= 0.0 || self.dt().is_nan() {
                return Err(IntegrationError::TimeStepNotSet(
                    time[0],
                    time[1],
                    format!("{self:?}"),
                ));
            } else {
                let num_steps = ((t_f - t_0) / self.dt()).ceil() as usize;
                t_sol = (0..num_steps)
                    .map(|step| t_0 + (step as Scalar) * self.dt())
                    .collect();
                t_sol.push(t_f);
            }
        } else {
            t_sol = time.iter().copied().collect();
        }
        let mut index = 0;
        let mut t = t_0;
        let mut dt;
        let mut t_trial;
        let mut y = initial_condition.clone();
        let mut y_sol = U::new();
        y_sol.push(initial_condition.clone());
        let mut dydt_sol = U::new();
        dydt_sol.push(function(t, &y.clone())?);
        let mut y_trial;
        while t < t_f {
            t_trial = t_sol[index + 1];
            dt = t_trial - t;
            y_trial = match solver.root(
                |y_trial: &Y| self.residual(&function, t, &y, t_trial, y_trial, dt),
                |y_trial: &Y| self.hessian(&jacobian, t, &y, t_trial, y_trial, dt),
                y.clone(),
                EqualityConstraint::None,
            ) {
                Ok(solution) => solution,
                Err(error) => {
                    return Err(IntegrationError::Upstream(
                        format!("{error}"),
                        format!("{self:?}"),
                    ));
                }
            };
            t = t_trial;
            y = y_trial;
            y_sol.push(y.clone());
            dydt_sol.push(function(t, &y)?);
            index += 1;
        }
        Ok((t_sol, y_sol, dydt_sol))
    }
    fn hessian(
        &self,
        jacobian: impl Fn(Scalar, &Y) -> Result<J, IntegrationError>,
        t: Scalar,
        y: &Y,
        t_trial: Scalar,
        y_trial: &Y,
        dt: Scalar,
    ) -> Result<J, String>;
}
