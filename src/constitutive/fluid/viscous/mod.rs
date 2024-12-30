//! Viscous constitutive models.

use super::*;

/// Required methods for viscous constitutive models.
pub trait Viscous<'a> {
    /// Returns the bulk viscosity.
    fn get_bulk_viscosity(&self) -> &Scalar;
    /// Returns the shear viscosity.
    fn get_shear_viscosity(&self) -> &Scalar;
}
