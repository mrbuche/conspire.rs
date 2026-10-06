pub(crate) mod stacked;
pub(crate) mod velocity_verlet;

use crate::{
    math::{
        Derivative, Differentiable, Quantity, Scalar, Tensor, TensorVec,
        integrate::{IntegrationError, Times},
    },
    units::Time,
};
use std::fmt::Debug;

/// Explicit integrators for dynamics, $`\ddot{x} = a(t, x, \dot{x})`$.
pub trait ExplicitDynamics<X, UX, UV, UA, T = Time>
where
    Self: Debug,
    X: Differentiable<T> + Tensor,
    Derivative<X, T>: Differentiable<T> + Tensor,
    UX: TensorVec<Item = X>,
    UV: TensorVec<Item = Derivative<X, T>>,
    UA: TensorVec<Item = Derivative<Derivative<X, T>, T>>,
{
    #[doc = include_str!("doc.md")]
    fn integrate(
        &self,
        function: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
        ) -> Result<Derivative<Derivative<X, T>, T>, String>,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
    ) -> Result<(Times<T>, UX, UV, UA), IntegrationError>;
    #[doc = include_str!("bounded.md")]
    fn integrate_bounded(
        &self,
        function: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
        ) -> Result<Derivative<Derivative<X, T>, T>, String>,
        bound: impl FnMut(Quantity<T>, &X, &Derivative<X, T>) -> Result<Quantity<T>, String>,
        safety: Scalar,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
    ) -> Result<(Times<T>, UX, UV, UA), IntegrationError>;
}
