#[cfg(test)]
mod test;

use crate::math::{
    Derivative, Differentiable, Quantity, Scalar, Tensor, TensorVec,
    integrate::{FixedStep, ImplicitDynamics, ImplicitDynamicsCoupled, IntegrationError, Times},
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
    fn time_grid<T>(&self, time: &[Quantity<T>]) -> Result<Times<T>, IntegrationError> {
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
        Ok(t_sol)
    }
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
        let t_sol = self.time_grid::<T>(time)?;
        let (t_0, t_f) = (t_sol[0], t_sol[t_sol.len() - 1]);
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

impl<X, Y, Z, F, J, UX, UV, UA, UY, UR, T>
    ImplicitDynamicsCoupled<X, Y, Z, F, J, UX, UV, UA, UY, UR, T> for Newmark
where
    X: Differentiable<T> + Tensor,
    Derivative<X, T>: Differentiable<T> + Tensor,
    Y: Differentiable<T> + Tensor,
    Z: Tensor,
    for<'a> &'a Derivative<X, T>: Mul<Quantity<T>, Output = X>,
    for<'a> &'a Derivative<Derivative<X, T>, T>: Mul<Quantity<T>, Output = Derivative<X, T>>,
    for<'a> &'a X: Div<Quantity<T>, Output = Derivative<X, T>>,
    for<'a> &'a X: Sub<&'a X, Output = X>,
    for<'a> &'a Derivative<X, T>: Div<Quantity<T>, Output = Derivative<Derivative<X, T>, T>>,
    for<'a> &'a Y: Sub<&'a Y, Output = Y>,
    for<'a> &'a Y: Div<Quantity<T>, Output = Derivative<Y, T>>,
    UX: TensorVec<Item = X>,
    UV: TensorVec<Item = Derivative<X, T>>,
    UA: TensorVec<Item = Derivative<Derivative<X, T>, T>>,
    UY: TensorVec<Item = Y>,
    UR: TensorVec<Item = Derivative<Y, T>>,
{
    fn integrate_coupled(
        &self,
        mut function: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
            &Y,
        )
            -> Result<(Derivative<Derivative<X, T>, T>, Derivative<Y, T>), String>,
        mut residual: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
            &Derivative<Derivative<X, T>, T>,
            &Y,
            &Derivative<Y, T>,
        ) -> Result<F, String>,
        mut tangent: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
            &Y,
            Quantity<T>,
            Scalar,
            Scalar,
        ) -> Result<J, String>,
        join: impl Fn(&X, &Y) -> Z,
        split: impl Fn(&Z) -> (X, Y),
        solver: impl RootFinding<F, J, Z>,
        equality_constraint: EqualityConstraint,
        sparse: Option<SparseSolver>,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
        initial_state: Y,
    ) -> Result<(Times<T>, UX, UV, UA, UY, UR), IntegrationError> {
        let t_sol = self.time_grid::<T>(time)?;
        let t_f = t_sol[t_sol.len() - 1];
        let (beta, gamma) = (self.beta, self.gamma);
        let mut index = 0;
        let mut t = t_sol[0];
        let mut x = initial_position;
        let mut v = initial_velocity;
        let mut y = initial_state;
        let (mut a, mut r) = function(t, &x, &v, &y)?;
        let mut x_sol = UX::new();
        x_sol.push(x.clone());
        let mut v_sol = UV::new();
        v_sol.push(v.clone());
        let mut a_sol = UA::new();
        a_sol.push(a.clone());
        let mut y_sol = UY::new();
        y_sol.push(y.clone());
        let mut r_sol = UR::new();
        r_sol.push(r.clone());
        while t < t_f {
            let t_next = t_sol[index + 1];
            let dt = t_next - t;
            let x_predictor = &(&a * (dt * (0.5 - beta)) + &v) * dt + &x;
            let v_predictor = &a * (dt * (1.0 - gamma)) + &v;
            let motion = |x_trial: &X| {
                let a_trial = &(&(x_trial - &x_predictor) / dt) / (dt * beta);
                let v_trial = &a_trial * (dt * gamma) + &v_predictor;
                (a_trial, v_trial)
            };
            let z_next = solver
                .root(
                    |z_trial: &Z| {
                        let (x_trial, y_trial) = split(z_trial);
                        let (a_trial, v_trial) = motion(&x_trial);
                        residual(
                            t_next,
                            &x_trial,
                            &v_trial,
                            &a_trial,
                            &y_trial,
                            &(&(&y_trial - &y) / dt),
                        )
                    },
                    |z_trial: &Z| {
                        let (x_trial, y_trial) = split(z_trial);
                        let (_, v_trial) = motion(&x_trial);
                        tangent(t_next, &x_trial, &v_trial, &y_trial, dt, beta, gamma)
                    },
                    join(&x_predictor, &y),
                    equality_constraint.clone(),
                    sparse.clone(),
                )
                .map_err(|error| IntegrationError::upstream(error, self))?;
            let (x_next, y_next) = split(&z_next);
            (a, v) = motion(&x_next);
            r = &(&y_next - &y) / dt;
            x = x_next;
            y = y_next;
            t = t_next;
            x_sol.push(x.clone());
            v_sol.push(v.clone());
            a_sol.push(a.clone());
            y_sol.push(y.clone());
            r_sol.push(r.clone());
            index += 1;
        }
        Ok((t_sol, x_sol, v_sol, a_sol, y_sol, r_sol))
    }
}
