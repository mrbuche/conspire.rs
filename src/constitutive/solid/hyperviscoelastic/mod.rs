//! Hyperviscoelastic solid constitutive models.
//!
//! ---
//!
#![doc = include_str!("doc.md")]

#[cfg(feature = "doc")]
pub mod doc;

#[cfg(test)]
pub mod test;

mod saint_venant_kirchhoff;

pub use saint_venant_kirchhoff::SaintVenantKirchhoff;

use super::{elastic_hyperviscous::ElasticHyperviscous, *};
use crate::{math::Quantity, units::EnergyDensity};

/// Required methods for hyperviscoelastic solid constitutive models.
pub trait Hyperviscoelastic
where
    Self: ElasticHyperviscous,
{
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a = a(\mathbf{F})
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError>;
}
