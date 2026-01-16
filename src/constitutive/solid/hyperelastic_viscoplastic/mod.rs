//! Hyperelastic-viscoplastic solid constitutive models.

mod hencky;
mod saint_venant_kirchhoff;

pub use hencky::Hencky;
pub use saint_venant_kirchhoff::SaintVenantKirchhoff;

use crate::{
    constitutive::{ConstitutiveError, solid::elastic_viscoplastic::ElasticViscoplastic},
    mechanics::{DeformationGradient, DeformationGradientPlastic, Scalar},
};

/// Required methods for hyperelastic-viscoplastic solid constitutive models.
pub trait HyperelasticViscoplastic
where
    Self: ElasticViscoplastic,
{
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a = a(\mathbf{F}_\mathrm{e})
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<Scalar, ConstitutiveError>;
}
