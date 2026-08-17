#[cfg(test)]
mod test;

use crate::{
    math::{
        Derivative, Differentiate, Quantity, Scalar, Tensor, TensorVec,
        integrate::{FixedStep, IntegrationError, OdeIntegrator, Times},
        optimize::{
            EqualityConstraint, FirstOrderRootFinding, LinearSolver, ZerothOrderRootFinding,
        },
    },
    units::Time,
};

pub(crate) mod backward_euler;
pub(crate) mod midpoint;
pub(crate) mod trapezoidal;

/// Implicit integrators for ordinary differential equations using zeroth-order root-finding.
pub trait ImplicitZerothOrder<Y, U, V, T = Time>
where
    Self: FixedStep<T> + OdeIntegrator<Y, U>,
    Y: Differentiate<T> + Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    #[doc = include_str!("doc.md")]
    fn integrate(
        &self,
        mut function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, IntegrationError>,
        time: &[Quantity<T>],
        initial_condition: Y,
        solver: impl ZerothOrderRootFinding<Y, Y>,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        let t_0 = time[0];
        let t_f = time[time.len() - 1];
        let mut t_sol: Times<T>;
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        } else if t_0 >= t_f {
            return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
        } else if time.len() == 2 {
            if self.dt() <= Quantity::default() || self.dt().is_nan() {
                return Err(IntegrationError::TimeStepNotSet(
                    time[0].value(),
                    time[1].value(),
                    format!("{self:?}"),
                ));
            } else {
                let max_steps = ((t_f - t_0).value() / self.dt().value()).ceil() as usize;
                t_sol = (0..max_steps)
                    .map(|step| t_0 + self.dt() * (step as Scalar))
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
        let mut dydt_sol = V::new();
        dydt_sol.push(function(t, &y.clone())?);
        let mut y_trial;
        while t < t_f {
            t_trial = t_sol[index + 1];
            dt = t_trial - t;
            y_trial = match solver.root(
                |y_trial: &Y| self.residual(&mut function, t, &y, t_trial, y_trial, dt),
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
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, IntegrationError>,
        t: Quantity<T>,
        y: &Y,
        t_trial: Quantity<T>,
        y_trial: &Y,
        dt: Quantity<T>,
    ) -> Result<Y, String>;
}

/// Implicit integrators for ordinary differential equations using first-order root-finding.
pub trait ImplicitFirstOrder<Y, J, U, V, T = Time>
where
    Self: ImplicitZerothOrder<Y, U, V, T>,
    Y: Differentiate<T> + Tensor,
    J: Differentiate<T> + Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    #[doc = include_str!("doc.md")]
    fn integrate(
        &self,
        mut function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, IntegrationError>,
        mut jacobian: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<J, T>, IntegrationError>,
        time: &[Quantity<T>],
        initial_condition: Y,
        solver: impl FirstOrderRootFinding<Y, J, Y>,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        let t_0 = time[0];
        let t_f = time[time.len() - 1];
        let mut t_sol: Times<T>;
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        } else if t_0 >= t_f {
            return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
        } else if time.len() == 2 {
            if self.dt() <= Quantity::default() || self.dt().is_nan() {
                return Err(IntegrationError::TimeStepNotSet(
                    time[0].value(),
                    time[1].value(),
                    format!("{self:?}"),
                ));
            } else {
                let max_steps = ((t_f - t_0).value() / self.dt().value()).ceil() as usize;
                t_sol = (0..max_steps)
                    .map(|step| t_0 + self.dt() * (step as Scalar))
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
        let mut dydt_sol = V::new();
        dydt_sol.push(function(t, &y.clone())?);
        let mut y_trial;
        while t < t_f {
            t_trial = t_sol[index + 1];
            dt = t_trial - t;
            y_trial = match solver.root(
                |y_trial: &Y| self.residual(&mut function, t, &y, t_trial, y_trial, dt),
                |y_trial: &Y| self.hessian(&mut jacobian, t, &y, t_trial, y_trial, dt),
                y.clone(),
                EqualityConstraint::None,
                LinearSolver::Dense,
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
        jacobian: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<J, T>, IntegrationError>,
        t: Quantity<T>,
        y: &Y,
        t_trial: Quantity<T>,
        y_trial: &Y,
        dt: Quantity<T>,
    ) -> Result<J, String>;
}
