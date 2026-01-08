//! Plastic fluid constitutive models.

use crate::math::Scalar;
use std::fmt::Debug;

/// Required methods for plastic fluid constitutive models.
pub trait Plastic
where
    Self: Clone + Debug,
{
    /// Returns the initial yield stress.
    fn initial_yield_stress(&self) -> Scalar;
    /// Returns the isotropic hardening slope.
    fn hardening_slope(&self) -> Scalar;
}
