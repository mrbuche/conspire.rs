pub(crate) mod stacked;
pub(crate) mod velocity_verlet;

use crate::{
    math::{
        Derivative, Differentiable, Quantity, Scalar, Tensor, TensorVec,
        integrate::{IntegrationError, Spectrum, Times},
    },
    units::Time,
};
use std::fmt::Debug;

/// Explicit integrators for dynamics.
/// ```math
/// \ddot{x} = a(t, x, \dot{x})
/// ```
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

/// Explicit integrators for dynamics coupled to a first-order state.
/// ```math
/// \ddot{x} = a(t, x, \dot{x}, y),\quad \dot{y} = f(t, x, \dot{x}, y)
/// ```
pub trait ExplicitDynamicsCoupled<X, Y, UX, UV, UA, UY, UR, T = Time>
where
    Self: Debug,
    X: Differentiable<T> + Tensor,
    Derivative<X, T>: Differentiable<T> + Tensor,
    Y: Differentiable<T> + Tensor,
    UX: TensorVec<Item = X>,
    UV: TensorVec<Item = Derivative<X, T>>,
    UA: TensorVec<Item = Derivative<Derivative<X, T>, T>>,
    UY: TensorVec<Item = Y>,
    UR: TensorVec<Item = Derivative<Y, T>>,
{
    /// Solves an initial value problem for dynamics coupled to a first-order state, given the
    /// acceleration and the rate of the state as a function of time, position, velocity, and
    /// state.
    ///
    /// The solution contains the positions, velocities, accelerations, states, and rates of the
    /// states at each time. The time step is that of the integrator when `time` holds only the
    /// initial and final times, and otherwise `time` is the sequence of steps.
    #[expect(clippy::type_complexity)]
    fn integrate_coupled(
        &self,
        function: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
            &Y,
        )
            -> Result<(Derivative<Derivative<X, T>, T>, Derivative<Y, T>), String>,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
        initial_state: Y,
    ) -> Result<(Times<T>, UX, UV, UA, UY, UR), IntegrationError>;
    /// Solves an initial value problem like [`integrate_coupled`](Self::integrate_coupled), subject to a
    /// stability bound.
    ///
    /// At each step, `bound` reports the spectrum of the coupled system at the current state,
    /// which is real for diffusion, imaginary for waves, and otherwise complex. The time step
    /// may use at most the fraction `safety` of the stability limit, where `safety` is in
    /// $`(0, 1]`$. A time step above it is an error.
    #[expect(clippy::type_complexity)]
    #[expect(clippy::too_many_arguments)]
    fn integrate_coupled_bounded(
        &self,
        function: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
            &Y,
        )
            -> Result<(Derivative<Derivative<X, T>, T>, Derivative<Y, T>), String>,
        bound: impl FnMut(Quantity<T>, &X, &Derivative<X, T>, &Y) -> Result<Spectrum<T>, String>,
        safety: Scalar,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
        initial_state: Y,
    ) -> Result<(Times<T>, UX, UV, UA, UY, UR), IntegrationError>;
}
