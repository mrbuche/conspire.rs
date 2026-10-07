#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        canonical::Canonical,
        fluid::plastic::{PlasticWork, RateIndependentPlastic},
        solid::{
            elastic::Elastic, hyperelastic::Hyperelastic, hyperelastic_plastic::HyperelasticPlastic,
        },
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

impl<C1, C2> PlasticWork for Canonical<C1, C2>
where
    C1: Elastic,
    C2: PlasticWork,
{
    fn plastic_work_density(
        &self,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        self.1.plastic_work_density(equivalent_plastic_strain)
    }
}
