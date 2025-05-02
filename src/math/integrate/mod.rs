#[cfg(test)]
mod test;

mod backward_euler;
mod bogacki_shampine;
mod dormand_prince;
mod verner_8;
mod verner_9;

pub use backward_euler::BackwardEuler;
pub use bogacki_shampine::BogackiShampine;
pub use dormand_prince::DormandPrince;
pub use verner_8::Verner8;
pub use verner_9::Verner9;

pub type Ode1be = BackwardEuler;
pub type Ode23 = BogackiShampine;
pub type Ode45 = DormandPrince;
pub type Ode78 = Verner8;
pub type Ode89 = Verner9;

// consider symplectic integrators for dynamics eventually

use super::{
    Tensor, TensorArray, TensorRank0, TensorVec, Vector, interpolate::InterpolateSolution,
};
use crate::defeat_message;
use std::{
    fmt::{self, Debug, Display, Formatter},
    ops::{Div, Mul, Sub},
};

/// Base trait for ordinary differential equation solvers.
pub trait OdeSolver<Y, U>
where
    Self: Debug,
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl<A, Y, U> OdeSolver<Y, U> for A
where
    A: Debug,
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
        time: &[TensorRank0],
        initial_condition: Y,
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
        time: &[TensorRank0],
        initial_condition: Y,
    ) -> Result<(Vector, U), IntegrationError>;
}

/// Possible errors encountered when integrating.
pub enum IntegrationError {
    InitialTimeNotLessThanFinalTime,
    LengthTimeLessThanTwo,
}

impl Debug for IntegrationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::InitialTimeNotLessThanFinalTime => {
                "\x1b[1;91mThe initial time must precede the final time.".to_string()
            }
            Self::LengthTimeLessThanTwo => {
                "\x1b[1;91mThe time must contain at least two entries.".to_string()
            }
        };
        write!(f, "\n{}\n\x1b[0;2;31m{}\x1b[0m\n", error, defeat_message())
    }
}

impl Display for IntegrationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
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
