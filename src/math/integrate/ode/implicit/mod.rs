use crate::math::{
    Scalar, Solution, Tensor, TensorArray, TensorVec, Vector,
    integrate::{IntegrationError, OdeSolver},
    interpolate::InterpolateSolution,
    optimize::{FirstOrderRootFinding, ZerothOrderRootFinding},
};
use std::ops::{Div, Mul, Sub};

pub mod backward_euler;

/// Zeroth-order implicit ordinary differential equation solvers.
pub trait ImplicitZerothOrder<Y, U>
where
    Self: InterpolateSolution<Y, U> + OdeSolver<Y, U>,
    Y: Solution,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    #[doc = include_str!("doc.md")]
    fn integrate(
        &self,
        function: impl Fn(Scalar, &Y) -> Result<Y, IntegrationError>,
        time: &[Scalar],
        initial_condition: Y,
        solver: impl ZerothOrderRootFinding<Y>,
    ) -> Result<(Vector, U, U), IntegrationError>;
}

/// First-order implicit ordinary differential equation solvers.
pub trait ImplicitFirstOrder<Y, J, U>
where
    Self: InterpolateSolution<Y, U> + OdeSolver<Y, U>,
    Y: Solution + Div<J, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    J: Tensor + TensorArray,
    U: TensorVec<Item = Y>,
{
    #[doc = include_str!("doc.md")]
    fn integrate(
        &self,
        function: impl Fn(Scalar, &Y) -> Result<Y, IntegrationError>,
        jacobian: impl Fn(Scalar, &Y) -> Result<J, IntegrationError>,
        time: &[Scalar],
        initial_condition: Y,
        solver: impl FirstOrderRootFinding<Y, J, Y>,
    ) -> Result<(Vector, U, U), IntegrationError>;
}
