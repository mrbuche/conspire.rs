#[cfg(test)]
mod test;

use crate::{
    math::{
        Derivative, Differentiable, Quantity, Scalar, Tensor, TensorVec,
        integrate::{IntegrationError, OdeIntegrator, StabilityInterval, Times},
    },
    units::Time,
};

pub(crate) mod fixed_step;
pub(crate) mod variable_step;

/// The fastest time scale of a right-hand side function, which bounds a stable time step.
///
/// The time scale is the reciprocal of the spectral radius of the Jacobian.
#[derive(Clone, Copy, Debug)]
pub enum Spectrum<T = Time> {
    /// The eigenvalues are real and negative, as in diffusion.
    Real(Quantity<T>),
    /// The eigenvalues are imaginary, as in waves.
    Imaginary(Quantity<T>),
}

impl<T> Spectrum<T> {
    pub(crate) fn limit(self, stability: StabilityInterval) -> Quantity<T> {
        match self {
            Self::Real(scale) => scale * stability.real,
            Self::Imaginary(scale) => scale * stability.imaginary,
        }
    }
}

pub(crate) fn check_safety(safety: Scalar) -> Result<(), IntegrationError> {
    if safety > 0.0 && safety <= 1.0 {
        Ok(())
    } else {
        Err(IntegrationError::InvalidSafetyFactor(safety))
    }
}

/// Explicit integrators for ordinary differential equations.
pub trait Explicit<Y, U, V, T = Time>
where
    Self: OdeIntegrator<Y, U>,
    Y: Differentiable<T> + Tensor,
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
    #[doc = include_str!("bounded.md")]
    fn integrate_bounded(
        &self,
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        bound: impl FnMut(Quantity<T>, &Y) -> Result<Spectrum<T>, String>,
        safety: Scalar,
        time: &[Quantity<T>],
        initial_condition: Y,
    ) -> Result<(Times<T>, U, V), IntegrationError>;
}
