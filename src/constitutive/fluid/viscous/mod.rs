//! Viscous constitutive models.

use super::*;

/// Required methods for viscous constitutive models.
pub trait Viscous<'a> {
    /// Returns the bulk viscosity.
    fn bulk_viscosity(&self) -> &Scalar;
    /// Returns the shear viscosity.
    fn shear_viscosity(&self) -> &Scalar;
}
