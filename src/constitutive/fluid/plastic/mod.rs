//! Plastic constitutive models.

use crate::math::Scalar;
use std::fmt::Debug;

/// Required methods for plastic constitutive models.
pub trait Plastic
where
    Self: Debug,
{
    /// Returns the initial yield stress.
    fn initial_yield_stress(&self) -> Scalar;
    /// Returns the isotropic hardening slope.
    fn hardening_slope(&self) -> Scalar;
}
