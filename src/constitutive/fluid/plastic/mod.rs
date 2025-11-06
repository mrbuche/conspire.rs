//! Plastic constitutive models.

use crate::math::Scalar;

/// Required methods for plastic constitutive models.
pub trait Plastic {
    /// Returns the initial yield stress.
    fn initial_yield_stress(&self) -> &Scalar;
    /// Returns the isotropic hardening slope.
    fn hardening_slope(&self) -> &Scalar;
}
