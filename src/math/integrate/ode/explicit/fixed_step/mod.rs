#[cfg(test)]
mod test;

use crate::math::unit::Time;
use crate::math::{
    Derivative, Differentiate, Quantity, Scalar, Tensor, TensorVec,
    integrate::{Explicit, FixedStep, IntegrationError, Times},
};

pub(crate) mod bogacki_shampine;
pub(crate) mod dormand_prince;
pub(crate) mod euler;
pub(crate) mod heun;
pub(crate) mod midpoint;
pub(crate) mod ralston;
pub(crate) mod verner_8;
pub(crate) mod verner_9;

/// Fixed-step explicit integrators for ordinary differential equations.
pub trait FixedStepExplicit<Y, U, V, T = Time>
where
    Self: Explicit<Y, U, V, T> + FixedStep<T>,
    Y: Differentiate<T> + Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    fn integrate_fixed_step(
        &self,
        mut function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        time: &[Quantity<T>],
        initial_condition: Y,
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
        let mut k = vec![Derivative::<Y, T>::default(); Self::SLOPES];
        k[0] = function(t, &initial_condition)?;
        let mut y = initial_condition.clone();
        let mut y_sol = U::new();
        y_sol.push(initial_condition.clone());
        let mut dydt_sol = V::new();
        dydt_sol.push(function(t, &y.clone())?);
        let mut y_trial = Y::default();
        while t < t_f {
            t_trial = t_sol[index + 1];
            dt = t_trial - t;
            if let Err(error) = self.step(&mut function, &y, t, dt, &mut k, &mut y_trial) {
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
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        y: &Y,
        t: Quantity<T>,
        dt: Quantity<T>,
        k: &mut [Derivative<Y, T>],
        y_trial: &mut Y,
    ) -> Result<(), String>;
}
