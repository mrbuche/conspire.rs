#[cfg(test)]
mod test;

use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::{IntegrationError, OdeSolver},
};

pub mod fixed_step;
pub mod variable_step;

/// Explicit ordinary differential equation solvers.
pub trait Explicit<Y, U>
where
    Self: OdeSolver<Y, U>,
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
    const SLOPES: usize;
    #[doc = include_str!("doc.md")]
    fn integrate(
        &self,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        time: &[Scalar],
        initial_condition: Y,
    ) -> Result<(Vector, U, U), IntegrationError>;
}
