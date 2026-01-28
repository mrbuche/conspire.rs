#[cfg(test)]
mod test;

use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::{Explicit, FixedStep, IntegrationError},
};

pub mod euler;
pub mod heun;
pub mod midpoint;
pub mod ralston;

/// Fixed-step explicit ordinary differential equation solvers.
pub trait FixedStepExplicit<Y, U>
where
    Self: Explicit<Y, U> + FixedStep,
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
    fn integrate_fixed_step(
        &self,
        mut function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        time: &[Scalar],
        initial_condition: Y,
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
        let mut k = vec![Y::default(); Self::SLOPES];
        k[0] = function(t, &initial_condition)?;
        let mut y = initial_condition.clone();
        let mut y_sol = U::new();
        y_sol.push(initial_condition.clone());
        let mut dydt_sol = U::new();
        dydt_sol.push(function(t, &y.clone())?);
        let mut y_trial = Y::default();
        while t < t_f {
            t_trial = t_sol[index + 1];
            dt = t_trial - t;
            if let Err(error) = self.step(&mut function, &mut y, &mut t, dt, &mut k, &mut y_trial) {
                return Err(IntegrationError::Upstream(error, format!("{self:?}")));
            } else {
                t += dt;
                y = y_trial.clone();
                y_sol.push(y.clone());
                dydt_sol.push(k[0].clone());
                index += 1;
            }
        }
        Ok((t_sol, y_sol, dydt_sol))
    }
    #[allow(clippy::too_many_arguments)]
    fn step(
        &self,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &mut Y,
        t: &mut Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
    ) -> Result<(), String>;
}
