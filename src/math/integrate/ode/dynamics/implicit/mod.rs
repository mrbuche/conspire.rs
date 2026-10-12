pub(crate) mod newmark;

use crate::{
    math::{
        Derivative, Differentiable, Quantity, Scalar, Tensor, TensorVec,
        integrate::{IntegrationError, Times},
        optimize::{EqualityConstraint, RootFinding},
        sparse::SparseSolver,
    },
    units::Time,
};
use std::fmt::Debug;

/// Implicit integrators for dynamics.
/// ```math
/// M\ddot{x} + f(t, x, \dot{x}) = 0
/// ```
pub trait ImplicitDynamics<X, F, J, UX, UV, UA, T = Time>
where
    Self: Debug,
    X: Differentiable<T> + Tensor,
    Derivative<X, T>: Differentiable<T> + Tensor,
    UX: TensorVec<Item = X>,
    UV: TensorVec<Item = Derivative<X, T>>,
    UA: TensorVec<Item = Derivative<Derivative<X, T>, T>>,
{
    #[doc = include_str!("doc.md")]
    #[expect(clippy::too_many_arguments)]
    fn integrate(
        &self,
        acceleration: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
        ) -> Result<Derivative<Derivative<X, T>, T>, String>,
        residual: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
            &Derivative<Derivative<X, T>, T>,
        ) -> Result<F, String>,
        tangent: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
            Quantity<T>,
            Scalar,
            Scalar,
        ) -> Result<J, String>,
        solver: impl RootFinding<F, J, X>,
        equality_constraint: EqualityConstraint,
        sparse: Option<SparseSolver>,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
    ) -> Result<(Times<T>, UX, UV, UA), IntegrationError>;
}
