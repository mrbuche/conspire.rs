#[cfg(test)]
mod test;

use crate::{
    math::{
        Derivative, Differentiable, Quantity, Tensor, TensorVec,
        integrate::{BackwardEuler, FixedStep, IntegrationError, Times, fixed_step_times},
        optimize::{EqualityConstraints, RootFinding},
        sparse::SparseSolver,
    },
    units::Time,
};
use std::{
    fmt::Debug,
    ops::{Div, Sub},
};

/// Implicit integrators for first-order systems.
/// ```math
/// C\dot{x} + f(t, x) = 0
/// ```
pub trait ImplicitRates<X, F, J, UX, UR, T = Time>
where
    Self: Debug,
    X: Differentiable<T> + Tensor,
    Derivative<X, T>: Tensor,
    UX: TensorVec<Item = X>,
    UR: TensorVec<Item = Derivative<X, T>>,
{
    #[doc = include_str!("doc.md")]
    #[expect(clippy::too_many_arguments)]
    fn integrate(
        &self,
        rate: impl FnMut(Quantity<T>, &X) -> Result<Derivative<X, T>, String>,
        residual: impl FnMut(Quantity<T>, &X, &Derivative<X, T>) -> Result<F, String>,
        tangent: impl FnMut(Quantity<T>, &X, &Derivative<X, T>, Quantity<T>) -> Result<J, String>,
        solver: impl RootFinding<F, J, X>,
        equality_constraint: impl Fn(Quantity<T>) -> EqualityConstraints,
        sparse: Option<SparseSolver>,
        time: &[Quantity<T>],
        initial_state: X,
    ) -> Result<(Times<T>, UX, UR), IntegrationError>;
}

impl<X, F, J, UX, UR, T> ImplicitRates<X, F, J, UX, UR, T> for BackwardEuler
where
    X: Differentiable<T> + Tensor,
    Derivative<X, T>: Tensor,
    for<'a> &'a X: Sub<&'a X, Output = X>,
    for<'a> &'a X: Div<Quantity<T>, Output = Derivative<X, T>>,
    UX: TensorVec<Item = X>,
    UR: TensorVec<Item = Derivative<X, T>>,
{
    fn integrate(
        &self,
        mut rate: impl FnMut(Quantity<T>, &X) -> Result<Derivative<X, T>, String>,
        mut residual: impl FnMut(Quantity<T>, &X, &Derivative<X, T>) -> Result<F, String>,
        mut tangent: impl FnMut(Quantity<T>, &X, &Derivative<X, T>, Quantity<T>) -> Result<J, String>,
        solver: impl RootFinding<F, J, X>,
        equality_constraint: impl Fn(Quantity<T>) -> EqualityConstraints,
        sparse: Option<SparseSolver>,
        time: &[Quantity<T>],
        initial_state: X,
    ) -> Result<(Times<T>, UX, UR), IntegrationError> {
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        }
        let t_0 = time[0];
        let t_f = time[time.len() - 1];
        let dt_fixed = FixedStep::<T>::dt(self);
        let t_sol: Times<T>;
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
                t_sol = fixed_step_times(t_0, t_f, dt_fixed);
            }
        } else {
            t_sol = time.iter().copied().collect();
        }
        let mut index = 0;
        let mut t = t_0;
        let mut x = initial_state;
        let mut x_sol = UX::new();
        x_sol.push(x.clone());
        let mut r_sol = UR::new();
        r_sol.push(rate(t, &x)?);
        while t < t_f {
            let t_next = t_sol[index + 1];
            let dt = t_next - t;
            let x_next = solver
                .root(
                    |x_trial: &X| residual(t_next, x_trial, &(&(x_trial - &x) / dt)),
                    |x_trial: &X| tangent(t_next, x_trial, &(&(x_trial - &x) / dt), dt),
                    x.clone(),
                    equality_constraint(t_next),
                    sparse.clone(),
                )
                .map_err(|error| IntegrationError::upstream(error, self))?;
            r_sol.push(&(&x_next - &x) / dt);
            x = x_next;
            t = t_next;
            x_sol.push(x.clone());
            index += 1;
        }
        Ok((t_sol, x_sol, r_sol))
    }
}
