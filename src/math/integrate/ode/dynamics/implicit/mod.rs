pub(crate) mod newmark;

use crate::{
    math::{
        Derivative, Differentiable, Quantity, Scalar, Tensor, TensorVec,
        integrate::{IntegrationError, Times},
        optimize::{EqualityConstraints, RootFinding},
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
        equality_constraint: EqualityConstraints,
        sparse: Option<SparseSolver>,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
    ) -> Result<(Times<T>, UX, UV, UA), IntegrationError>;
}

/// Implicit integrators for dynamics coupled to a first-order state.
/// ```math
/// M\ddot{x} + f(t, x, \dot{x}, y) = 0,\quad C\dot{y} + g(t, x, \dot{x}, y) = 0
/// ```
pub trait ImplicitDynamicsCoupled<X, Y, Z, F, J, UX, UV, UA, UY, UR, T = Time>
where
    Self: Debug,
    X: Differentiable<T> + Tensor,
    Derivative<X, T>: Differentiable<T> + Tensor,
    Y: Differentiable<T> + Tensor,
    Z: Tensor,
    UX: TensorVec<Item = X>,
    UV: TensorVec<Item = Derivative<X, T>>,
    UA: TensorVec<Item = Derivative<Derivative<X, T>, T>>,
    UY: TensorVec<Item = Y>,
    UR: TensorVec<Item = Derivative<Y, T>>,
{
    /// Solves an initial value problem for dynamics coupled to a first-order state, given the
    /// residual of the equations of motion and of the state as a function of time, position,
    /// velocity, acceleration, state, and rate of the state, and its tangent with respect to the
    /// position and the state.
    ///
    /// The position and the state are solved for together as one unknown, which `join` assembles
    /// from them and `split` separates into them. The velocity and acceleration follow from the
    /// position as in the uncoupled method, and the rate of the state from the backward
    /// difference of the state, so that the velocity is that of the method and not a difference
    /// of the positions. The acceleration and rate functions are only evaluated for the initial
    /// ones. The tangent is handed the time step and the parameters $`\beta`$ and $`\gamma`$ of
    /// the method, from which the weights of the mass, the velocity, and the capacity in the
    /// effective tangent follow. The solution contains the positions, velocities, accelerations,
    /// states, and rates of the states at each time.
    #[expect(clippy::type_complexity)]
    #[expect(clippy::too_many_arguments)]
    fn integrate_coupled(
        &self,
        function: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
            &Y,
        )
            -> Result<(Derivative<Derivative<X, T>, T>, Derivative<Y, T>), String>,
        residual: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
            &Derivative<Derivative<X, T>, T>,
            &Y,
            &Derivative<Y, T>,
        ) -> Result<F, String>,
        tangent: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
            &Y,
            Quantity<T>,
            Scalar,
            Scalar,
        ) -> Result<J, String>,
        join: impl Fn(&X, &Y) -> Z,
        split: impl Fn(&Z) -> (X, Y),
        solver: impl RootFinding<F, J, Z>,
        equality_constraint: EqualityConstraints,
        sparse: Option<SparseSolver>,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
        initial_state: Y,
    ) -> Result<(Times<T>, UX, UV, UA, UY, UR), IntegrationError>;
}
