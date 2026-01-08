//! Viscous fluid constitutive models.

use crate::math::Scalar;

/// Required methods for viscous fluid constitutive models.
pub trait Viscous {
    /// Returns the bulk viscosity.
    fn bulk_viscosity(&self) -> Scalar;
    /// Returns the shear viscosity.
    fn shear_viscosity(&self) -> Scalar;
}
