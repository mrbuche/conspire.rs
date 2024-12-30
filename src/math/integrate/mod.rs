#[cfg(test)]
mod test;

mod ode1be;
mod ode23;

// Explicit, six-stage, fifth-order, variable-step, Runge-Kutta method ([Dormand and Prince, 1980](https://doi.org/10.1016/0771-050X(80)90013-3)).
// mod ode45;

pub use ode1be::Ode1be;
pub use ode23::Ode23;

use super::{Tensor, TensorArray, TensorRank0, TensorRank0List};
use crate::get_defeat_message;
use std::{
    fmt,
    iter::Peekable,
    ops::{Div, Mul, Sub},
};

type EvalTimes<const W: usize> = Peekable<std::array::IntoIter<TensorRank0, W>>;

/// Base trait for ordinary differential equation solvers.
pub trait OdeSolver<Y, U, const W: usize>
where
    Self: fmt::Debug,
    Y: Tensor,
    U: Tensor<Item = Y>,
{
    /// Setup for ordinary differential equation solvers.
    fn setup<'a>(
        &'a self,
        initial_time: TensorRank0,
        initial_condition: Y,
        evaluation_times: &TensorRank0List<W>,
        solution: &'a mut U,
    ) -> Result<
        (
            EvalTimes<W>,
            TensorRank0,
            TensorRank0,
            Y,
            impl Iterator<Item = &'a mut <U as Tensor>::Item>,
        ),
        IntegrationError<W>,
    >
    where
        Y: 'a,
    {
        let dt;
        let y = initial_condition.copy();
        let t = initial_time.copy();
        let mut eval_times = evaluation_times.0.into_iter().peekable();
        let mut y_sol = solution.iter_mut();
        for check_times in evaluation_times.0.windows(2) {
            if check_times[1] - check_times[0] <= 0.0 {
                return Err(IntegrationError::EvaluationTimesNotStrictlyIncreasing(
                    evaluation_times.copy(),
                    format!("{:?}", &self),
                ));
            }
        }
        if eval_times.next_if_eq(&initial_time).is_some() {
            if W == 1 {
                return Err(IntegrationError::EvaluationTimesNoFinalTime(
                    evaluation_times.copy(),
                    format!("{:?}", &self),
                ));
            } else {
                dt = eval_times.peek().ok_or("not ok")? - initial_time;
                *y_sol.next().ok_or("not ok")? = initial_condition;
            }
        } else if eval_times.peek().ok_or("not ok")? > &initial_time {
            dt = eval_times.peek().ok_or("not ok")? - initial_time;
        } else {
            return Err(IntegrationError::EvaluationTimesPrecedeInitialTime(
                evaluation_times.copy(),
                initial_time,
                format!("{:?}", &self),
            ));
        };
        Ok((eval_times, dt, t, y, y_sol))
    }
}

impl<A, Y, U, const W: usize> OdeSolver<Y, U, W> for A
where
    A: std::fmt::Debug,
    Y: Tensor,
    U: Tensor<Item = Y>,
{
}

/// Base trait for explicit ordinary differential equation solvers.
pub trait Explicit<Y, U, const W: usize>: OdeSolver<Y, U, W>
where
    Y: Tensor,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    U: Tensor<Item = Y> + TensorArray,
{
    /// Solves an initial value problem by explicitly integrating a system of ordinary differential equations.
    ///
    /// ```math
    /// \frac{dy}{dt} = f(t, y),\quad y(t_0) = y_0
    /// ```
    fn integrate(
        &self,
        function: impl Fn(&TensorRank0, &Y) -> Y,
        initial_time: TensorRank0,
        initial_condition: Y,
        evaluation_times: &TensorRank0List<W>,
    ) -> Result<U, IntegrationError<W>>;
}

/// Base trait for implicit ordinary differential equation solvers.
pub trait Implicit<Y, J, U, const W: usize>: OdeSolver<Y, U, W>
where
    Y: Tensor + Div<J, Output = Y>,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    J: Tensor + TensorArray,
    U: Tensor<Item = Y> + TensorArray,
{
    /// Solves an initial value problem by implicitly integrating a system of ordinary differential equations.
    ///
    /// ```math
    /// \frac{dy}{dt} = f(t, y),\quad y(t_0) = y_0,\quad \frac{\partial f}{\partial y} = J(t, y)
    /// ```
    fn integrate(
        &self,
        function: impl Fn(&TensorRank0, &Y) -> Y,
        jacobian: impl Fn(&TensorRank0, &Y) -> J,
        initial_time: TensorRank0,
        initial_condition: Y,
        evaluation_times: &TensorRank0List<W>,
    ) -> Result<U, IntegrationError<W>>;
}

/// Possible errors encountered when integrating.
pub enum IntegrationError<const W: usize> {
    EvaluationTimesNoFinalTime(TensorRank0List<W>, String),
    EvaluationTimesNotStrictlyIncreasing(TensorRank0List<W>, String),
    EvaluationTimesPrecedeInitialTime(TensorRank0List<W>, TensorRank0, String),
}

impl<const W: usize> From<&str> for IntegrationError<W> {
    fn from(string: &str) -> Self {
        todo!("{}", string)
    }
}

impl<const W: usize> fmt::Debug for IntegrationError<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::EvaluationTimesNoFinalTime(evaluation_times, integrator) => {
                format!(
                    "\x1b[1;91mEvaluation times must include a final time.\x1b[0;91m\n\
                     From evaluation times: {}.\n\
                     In integrator: {}.",
                    evaluation_times, integrator
                )
            }
            Self::EvaluationTimesNotStrictlyIncreasing(evaluation_times, integrator) => {
                format!(
                    "\x1b[1;91mEvaluation times must be strictly increasing.\x1b[0;91m\n\
                     From evaluation times: {}.\n\
                     In integrator: {}.",
                    evaluation_times, integrator
                )
            }
            Self::EvaluationTimesPrecedeInitialTime(evaluation_times, initial_time, integrator) => {
                format!(
                    "\x1b[1;91mEvaluation times precede the initial time.\x1b[0;91m\n\
                     From evaluation times: {}.\n\
                     With initial time: {}.\n\
                     In integrator: {}.",
                    evaluation_times, initial_time, integrator
                )
            }
        };
        write!(
            f,
            "\n{}\n\x1b[0;2;31m{}\x1b[0m\n",
            error,
            get_defeat_message()
        )
    }
}

impl<const W: usize> fmt::Display for IntegrationError<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::EvaluationTimesNoFinalTime(evaluation_times, integrator) => {
                format!(
                    "\x1b[1;91mEvaluation times must include a final time.\x1b[0;91m\n\
                     From evaluation times: {}.\n\
                     In integrator: {}.",
                    evaluation_times, integrator
                )
            }
            Self::EvaluationTimesNotStrictlyIncreasing(evaluation_times, integrator) => {
                format!(
                    "\x1b[1;91mEvaluation times must be strictly increasing.\x1b[0;91m\n\
                     From evaluation times: {}.\n\
                     In integrator: {}.",
                    evaluation_times, integrator
                )
            }
            Self::EvaluationTimesPrecedeInitialTime(evaluation_times, initial_time, integrator) => {
                format!(
                    "\x1b[1;91mEvaluation times precede the initial time.\x1b[0;91m\n\
                     From evaluation times: {}.\n\
                     With initial time: {}.\n\
                     In integrator: {}.",
                    evaluation_times, initial_time, integrator
                )
            }
        };
        write!(f, "{}\x1b[0m", error)
    }
}
