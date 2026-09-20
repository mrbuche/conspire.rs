//! Hyperelastic-plastic solid constitutive models.

mod canonical;

use crate::{
    constitutive::{ConstitutiveError, solid::elastic_plastic::ElasticPlastic},
    math::Quantity,
    mechanics::{DeformationGradient, DeformationGradientPlastic},
    units::EnergyDensity,
};

/// Required methods for hyperelastic-plastic solid constitutive models.
pub trait HyperelasticPlastic
where
    Self: ElasticPlastic,
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
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError>;
}
