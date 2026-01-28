use crate::math::{Scalar, Tensor, TensorVec};
use std::fmt::Debug;

pub mod explicit;
pub mod implicit;

/// Ordinary differential equation solvers.
pub trait OdeSolver<Y, U>
where
    Self: Debug,
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

/// Fixed-step ordinary differential equation solvers.
pub trait FixedStep {
    /// Returns the time step.
    fn dt(&self) -> Scalar;
}

/// Variable-step ordinary differential equation solvers.
pub trait VariableStep {
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
    fn dt_min(&self) -> Scalar;
}
