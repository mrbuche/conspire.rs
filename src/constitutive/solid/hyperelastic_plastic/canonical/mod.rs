#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        canonical::Canonical,
        fluid::plastic::RateIndependentPlastic,
        solid::{hyperelastic::Hyperelastic, hyperelastic_plastic::HyperelasticPlastic},
    },
    math::Quantity,
    mechanics::{DeformationGradient, DeformationGradientPlastic},
    units::EnergyDensity,
};

impl<C1, C2> HyperelasticPlastic for Canonical<C1, C2>
where
    C1: Hyperelastic,
    C2: RateIndependentPlastic,
{
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        self.0
            .helmholtz_free_energy_density(&deformation_gradient_e.into())
    }
}
