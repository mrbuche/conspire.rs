#[cfg(test)]
mod test;

use crate::math::{
    Derivative, Differentiate, Quantity, Tensor, TensorVec, Time,
    integrate::{IntegrationError, OdeIntegrator, Times},
};

pub(crate) mod fixed_step;
pub(crate) mod variable_step;

/// Explicit integrators for ordinary differential equations.
pub trait Explicit<Y, U, V, T = Time>
where
    Self: OdeIntegrator<Y, U>,
    Y: Differentiate<T> + Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    const SLOPES: usize;
    #[doc = include_str!("doc.md")]
    fn integrate(
        &self,
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        time: &[Quantity<T>],
        initial_condition: Y,
    ) -> Result<(Times<T>, U, V), IntegrationError>;
}
