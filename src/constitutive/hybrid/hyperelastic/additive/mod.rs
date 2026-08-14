#[cfg(test)]
mod test;

use crate::math::Quantity;
use crate::{
    constitutive::{ConstitutiveError, hybrid::ElasticAdditive, solid::hyperelastic::Hyperelastic},
    mechanics::DeformationGradient,
    units::EnergyDensity,
};

impl<C1, C2> Hyperelastic for ElasticAdditive<C1, C2>
where
    C1: Hyperelastic,
    C2: Hyperelastic,
{
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a(\mathbf{F}) = a_1(\mathbf{F}) + a_2(\mathbf{F})
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        Ok(self.0.helmholtz_free_energy_density(deformation_gradient)?
            + self.1.helmholtz_free_energy_density(deformation_gradient)?)
    }
}
