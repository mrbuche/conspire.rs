#[cfg(test)]
mod test;

use crate::math::{
    Derivative, Differentiable, Quantity, Scalar, Tensor, TensorVec,
    integrate::{FixedStep, ImplicitDynamics, IntegrationError, Times},
    optimize::{EqualityConstraint, RootFinding},
    sparse::SparseSolver,
};
use std::ops::{Div, Mul, Sub};

#[doc = include_str!("doc.md")]
#[derive(Debug)]
pub struct Newmark {
    /// Fixed value for the time step.
    dt: Scalar,
    beta: Scalar,
    gamma: Scalar,
}

impl Default for Newmark {
    fn default() -> Self {
        Self {
            dt: Scalar::NAN,
            beta: 0.25,
            gamma: 0.5,
        }
    }
}

impl<T> FixedStep<T> for Newmark {
    fn dt(&self) -> Quantity<T> {
        Quantity::new(self.dt)
    }
}

impl Newmark {
    /// Constructs the average acceleration method with a fixed time step.
    pub fn new<T>(dt: Quantity<T>) -> Self {
        Self {
            dt: dt.value(),
            ..Default::default()
        }
    }
    /// Constructs the method with a fixed time step and the given parameters.
    ///
    /// Only unconditionally stable parameters are accepted, which satisfy
    /// $`2\beta \ge \gamma \ge 1/2`$.
    pub fn with_parameters<T>(
        dt: Quantity<T>,
        beta: Scalar,
        gamma: Scalar,
    ) -> Result<Self, IntegrationError> {
        if gamma >= 0.5 && 2.0 * beta >= gamma {
            Ok(Self {
                dt: dt.value(),
                beta,
                gamma,
            })
        } else {
            Err(IntegrationError::InvalidNewmarkParameters(beta, gamma))
        }
    }
}

impl<X, F, J, UX, UV, UA, T> ImplicitDynamics<X, F, J, UX, UV, UA, T> for Newmark
where
    X: Differentiable<T> + Tensor,
    Derivative<X, T>: Differentiable<T> + Tensor,
    for<'a> &'a Derivative<X, T>: Mul<Quantity<T>, Output = X>,
    for<'a> &'a Derivative<Derivative<X, T>, T>: Mul<Quantity<T>, Output = Derivative<X, T>>,
    for<'a> &'a X: Div<Quantity<T>, Output = Derivative<X, T>>,
    for<'a> &'a X: Sub<&'a X, Output = X>,
    for<'a> &'a Derivative<X, T>: Div<Quantity<T>, Output = Derivative<Derivative<X, T>, T>>,
    UX: TensorVec<Item = X>,
    UV: TensorVec<Item = Derivative<X, T>>,
    UA: TensorVec<Item = Derivative<Derivative<X, T>, T>>,
{
    fn integrate(
        &self,
        mut acceleration: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
        ) -> Result<Derivative<Derivative<X, T>, T>, String>,
        mut residual: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
            &Derivative<Derivative<X, T>, T>,
        ) -> Result<F, String>,
        mut tangent: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
            Quantity<T>,
            Scalar,
            Scalar,
        ) -> Result<J, String>,
        solver: impl RootFinding<F, J, X>,
        equality_constraint: EqualityConstraint,
        sparse: Option<SparseSolver>,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
    ) -> Result<(Times<T>, UX, UV, UA), IntegrationError> {
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
        let (beta, gamma) = (self.beta, self.gamma);
        let mut index = 0;
        let mut t = t_0;
        let mut x = initial_position;
        let mut v = initial_velocity;
        let mut a = acceleration(t, &x, &v)?;
        let mut x_sol = UX::new();
        x_sol.push(x.clone());
        let mut v_sol = UV::new();
        v_sol.push(v.clone());
        let mut a_sol = UA::new();
        a_sol.push(a.clone());
        while t < t_f {
            let t_next = t_sol[index + 1];
            let dt = t_next - t;
            let x_predictor = &(&a * (dt * (0.5 - beta)) + &v) * dt + &x;
            let v_predictor = &a * (dt * (1.0 - gamma)) + &v;
            let x_next = solver
                .root(
                    |x_trial: &X| {
                        let a_trial = &(&(x_trial - &x_predictor) / dt) / (dt * beta);
                        let v_trial = &a_trial * (dt * gamma) + &v_predictor;
                        residual(t_next, x_trial, &v_trial, &a_trial)
                    },
                    |x_trial: &X| {
                        let a_trial = &(&(x_trial - &x_predictor) / dt) / (dt * beta);
                        let v_trial = &a_trial * (dt * gamma) + &v_predictor;
                        tangent(t_next, x_trial, &v_trial, dt, beta, gamma)
                    },
                    x_predictor.clone(),
                    equality_constraint.clone(),
                    sparse.clone(),
                )
                .map_err(|error| IntegrationError::upstream(error, self))?;
            a = &(&(&x_next - &x_predictor) / dt) / (dt * beta);
            v = &a * (dt * gamma) + &v_predictor;
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
