#[cfg(test)]
mod test;

use crate::math::{
    Derivative, Differentiable, Quantity, Scalar, Tensor, TensorVec,
    integrate::{ExplicitDynamics, FixedStep, IntegrationError, Times},
};
use std::ops::Mul;

/// The extent of the stability region of the method along the imaginary axis.
const EXTENT: Scalar = 2.0;

#[doc = include_str!("doc.md")]
#[derive(Debug, Default)]
pub struct VelocityVerlet {
    /// Fixed value for the time step.
    dt: Scalar,
}

impl<T> FixedStep<T> for VelocityVerlet {
    fn dt(&self) -> Quantity<T> {
        Quantity::new(self.dt)
    }
}

impl VelocityVerlet {
    fn integrate_checked<X, UX, UV, UA, T>(
        &self,
        mut function: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
        ) -> Result<Derivative<Derivative<X, T>, T>, String>,
        mut check: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
            Quantity<T>,
        ) -> Result<(), IntegrationError>,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
    ) -> Result<(Times<T>, UX, UV, UA), IntegrationError>
    where
        X: Differentiable<T> + Tensor,
        Derivative<X, T>: Differentiable<T> + Tensor,
        for<'a> &'a Derivative<X, T>: Mul<Quantity<T>, Output = X>,
        for<'a> &'a Derivative<Derivative<X, T>, T>: Mul<Quantity<T>, Output = Derivative<X, T>>,
        UX: TensorVec<Item = X>,
        UV: TensorVec<Item = Derivative<X, T>>,
        UA: TensorVec<Item = Derivative<Derivative<X, T>, T>>,
    {
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        }
        let t_0 = time[0];
        let t_f = time[time.len() - 1];
        let dt_fixed = FixedStep::<T>::dt(self);
        let mut t_sol: Times<T>;
        if t_0 >= t_f {
            return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
        } else if time.len() == 2 {
            if dt_fixed <= Quantity::default() || dt_fixed.is_nan() {
                return Err(IntegrationError::TimeStepNotSet(
                    time[0].value(),
                    time[1].value(),
                    format!("{self:?}"),
                ));
            } else {
                let max_steps = ((t_f - t_0).value() / dt_fixed.value()).ceil() as usize;
                t_sol = (0..max_steps)
                    .map(|step| t_0 + dt_fixed * (step as Scalar))
                    .collect();
                t_sol.push(t_f);
            }
        } else {
            t_sol = time.iter().copied().collect();
        }
        let mut index = 0;
        let mut t = t_0;
        let mut x = initial_position;
        let mut v = initial_velocity;
        let mut a = function(t, &x, &v)?;
        let mut x_sol = UX::new();
        x_sol.push(x.clone());
        let mut v_sol = UV::new();
        v_sol.push(v.clone());
        let mut a_sol = UA::new();
        a_sol.push(a.clone());
        while t < t_f {
            let t_next = t_sol[index + 1];
            let dt = t_next - t;
            check(t, &x, &v, dt)?;
            let v_half = &a * (dt * 0.5) + &v;
            let x_next = &v_half * dt + &x;
            a = function(t_next, &x_next, &v_half)
                .map_err(|error| IntegrationError::upstream(error, self))?;
            v = &a * (dt * 0.5) + &v_half;
            x = x_next;
            t = t_next;
            x_sol.push(x.clone());
            v_sol.push(v.clone());
            a_sol.push(a.clone());
            index += 1;
        }
        Ok((t_sol, x_sol, v_sol, a_sol))
    }
}

impl<X, UX, UV, UA, T> ExplicitDynamics<X, UX, UV, UA, T> for VelocityVerlet
where
    X: Differentiable<T> + Tensor,
    Derivative<X, T>: Differentiable<T> + Tensor,
    for<'a> &'a Derivative<X, T>: Mul<Quantity<T>, Output = X>,
    for<'a> &'a Derivative<Derivative<X, T>, T>: Mul<Quantity<T>, Output = Derivative<X, T>>,
    UX: TensorVec<Item = X>,
    UV: TensorVec<Item = Derivative<X, T>>,
    UA: TensorVec<Item = Derivative<Derivative<X, T>, T>>,
{
    fn integrate(
        &self,
        function: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
        ) -> Result<Derivative<Derivative<X, T>, T>, String>,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
    ) -> Result<(Times<T>, UX, UV, UA), IntegrationError> {
        self.integrate_checked(
            function,
            |_, _, _, _| Ok(()),
            time,
            initial_position,
            initial_velocity,
        )
    }
    fn integrate_bounded(
        &self,
        function: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
        ) -> Result<Derivative<Derivative<X, T>, T>, String>,
        mut bound: impl FnMut(Quantity<T>, &X, &Derivative<X, T>) -> Result<Quantity<T>, String>,
        safety: Scalar,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
    ) -> Result<(Times<T>, UX, UV, UA), IntegrationError> {
        if !(safety > 0.0 && safety <= 1.0) {
            return Err(IntegrationError::InvalidSafetyFactor(safety));
        }
        self.integrate_checked(
            function,
            |t, x, v, dt| {
                let limit = bound(t, x, v)? * (EXTENT * safety);
                if dt > limit {
                    Err(IntegrationError::UnstableTimeStep(
                        dt.value(),
                        limit.value(),
                        format!("{self:?}"),
                    ))
                } else {
                    Ok(())
                }
            },
            time,
            initial_position,
            initial_velocity,
        )
    }
}
