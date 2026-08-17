use crate::math::{Norm, Quantity, Scalar, Tensor, TensorVec};
use crate::units::Time;
use std::fmt::Debug;

pub(super) mod explicit;
pub(super) mod implicit;

/// Integrators for ordinary differential equations.
pub trait OdeIntegrator<Y, U>
where
    Self: Debug,
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

/// Fixed-step integrators for ordinary differential equations.
pub trait FixedStep<T = Time> {
    /// Returns the time step.
    fn dt(&self) -> Quantity<T>;
}

/// Variable-step integrators for ordinary differential equations.
pub trait VariableStep<T = Time> {
    /// Returns the absolute error tolerance.
    fn abs_tol(&self) -> Scalar;
    /// Returns the relative error tolerance.
    fn rel_tol(&self) -> Scalar;
    /// Returns the multiplier for adaptive time steps.
    fn dt_beta(&self) -> Scalar;
    /// Returns the exponent for adaptive time steps.
    fn dt_expn(&self) -> Scalar;
    /// Returns the cut back factor for function errors.
    fn dt_cut(&self) -> Scalar;
    /// Returns the minimum value for the time step.
    fn dt_min(&self) -> Quantity<T>;
    /// Returns the norm type for error evaluation.
    fn error_norm(&self) -> &Norm;
}
