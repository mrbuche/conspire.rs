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
    Solution, Tensor, TensorArray, TensorRank0, TensorVec, TestError, Vector,
    interpolate::InterpolateSolution,
    optimize::{FirstOrderRootFinding, OptimizationError, ZerothOrderRootFinding},
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
    Y: Tensor,
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
        function: impl FnMut(TensorRank0, &Y) -> Result<Y, IntegrationError>,
        time: &[TensorRank0],
        initial_condition: Y,
    ) -> Result<(Vector, U, U), IntegrationError>;
}

/// Base trait for zeroth-order implicit ordinary differential equation solvers.
pub trait ImplicitZerothOrder<Y, U>: OdeSolver<Y, U>
where
    Self: InterpolateSolution<Y, U>,
    Y: Solution,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    /// Solves an initial value problem by implicitly integrating a system of ordinary differential equations.
    ///
    /// ```math
    /// \frac{dy}{dt} = f(t, y),\quad y(t_0) = y_0,\quad \frac{\partial f}{\partial y} = J(t, y)
    /// ```
    fn integrate(
        &self,
        function: impl Fn(TensorRank0, &Y) -> Result<Y, IntegrationError>,
        time: &[TensorRank0],
        initial_condition: Y,
        solver: impl ZerothOrderRootFinding<Y>,
    ) -> Result<(Vector, U, U), IntegrationError>;
}

/// Base trait for first-order implicit ordinary differential equation solvers.
pub trait ImplicitFirstOrder<Y, J, U>: OdeSolver<Y, U>
where
    Self: InterpolateSolution<Y, U>,
    Y: Solution + Div<J, Output = Y>,
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
        function: impl Fn(TensorRank0, &Y) -> Result<Y, IntegrationError>,
        jacobian: impl Fn(TensorRank0, &Y) -> Result<J, IntegrationError>,
        time: &[TensorRank0],
        initial_condition: Y,
        solver: impl FirstOrderRootFinding<Y, J, Y>,
    ) -> Result<(Vector, U, U), IntegrationError>;
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
        write!(f, "{error}\x1b[0m")
    }
}

impl From<OptimizationError> for IntegrationError {
    fn from(_error: OptimizationError) -> Self {
        todo!()
    }
}

impl From<IntegrationError> for String {
    fn from(error: IntegrationError) -> Self {
        format!("{}", error)
    }
}

impl From<IntegrationError> for TestError {
    fn from(error: IntegrationError) -> Self {
        TestError {
            message: error.to_string(),
        }
    }
}
