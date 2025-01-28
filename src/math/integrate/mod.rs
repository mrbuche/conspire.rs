#[cfg(test)]
mod test;

mod ode1be;
mod ode23;
mod ode45;

pub use ode1be::Ode1be;
pub use ode23::Ode23;
pub use ode45::Ode45;

use super::{
    interpolate::InterpolateSolution, Tensor, TensorArray, TensorRank0, TensorVec, Vector,
};
use crate::get_defeat_message;
use std::{
    fmt,
    ops::{Div, Mul, Sub},
};

/// Base trait for ordinary differential equation solvers.
pub trait OdeSolver<Y, U>
where
    Self: fmt::Debug,
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl<A, Y, U> OdeSolver<Y, U> for A
where
    A: std::fmt::Debug,
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

/// Base trait for explicit ordinary differential equation solvers.
pub trait Explicit<Y, U>: OdeSolver<Y, U>
where
    Self: InterpolateSolution<Y, U>,
    Y: Tensor + TensorArray,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
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
        time: &[TensorRank0],
    ) -> Result<(Vector, U), IntegrationError>;
}

/// Base trait for implicit ordinary differential equation solvers.
pub trait Implicit<Y, J, U>: OdeSolver<Y, U>
where
    Self: InterpolateSolution<Y, U>,
    Y: Tensor + TensorArray + Div<J, Output = Y>,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    J: Tensor + TensorArray,
    U: TensorVec<Item = Y>,
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
        time: &[TensorRank0],
    ) -> Result<(Vector, U), IntegrationError>;
}

/// Possible errors encountered when integrating.
pub enum IntegrationError {
    InitialTimeNotLessThanFinalTime,
    LengthTimeLessThanTwo,
}

impl From<&str> for IntegrationError {
    fn from(string: &str) -> Self {
        todo!("{}", string)
    }
}

impl fmt::Debug for IntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::InitialTimeNotLessThanFinalTime => {
                "\x1b[1;91mThe initial time must precede the final time.".to_string()
            }
            Self::LengthTimeLessThanTwo => {
                "\x1b[1;91mThe time must contain at least two entries.".to_string()
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

impl fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::InitialTimeNotLessThanFinalTime => {
                "\x1b[1;91mThe initial time must precede the final time.".to_string()
            }
            Self::LengthTimeLessThanTwo => {
                "\x1b[1;91mThe time must contain at least two entries.".to_string()
            }
        };
        write!(f, "{}\x1b[0m", error)
    }
}
