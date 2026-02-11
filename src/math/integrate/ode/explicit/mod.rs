#[cfg(test)]
mod test;

use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::{IntegrationError, OdeIntegrator},
};

pub mod fixed_step;
pub mod variable_step;

/// Explicit integrators for ordinary differential equations.
pub trait Explicit<Y, U>
where
    Self: OdeIntegrator<Y, U>,
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
