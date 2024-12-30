//! Thermohyperelastic constitutive models.

#[cfg(test)]
pub mod test;

mod saint_venant_kirchoff;

pub use saint_venant_kirchoff::SaintVenantKirchoff;

use super::{thermoelastic::Thermoelastic, *};

/// Required methods for thermohyperelastic constitutive models.
pub trait Thermohyperelastic<'a>
where
    Self: Thermoelastic<'a>,
{
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a = a(\mathbf{F},T)
    /// ```
    fn calculate_helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<Scalar, ConstitutiveError>;
}
