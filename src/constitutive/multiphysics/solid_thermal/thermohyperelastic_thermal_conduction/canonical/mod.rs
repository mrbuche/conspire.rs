#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        multiphysics::{Canonical, SolidThermal},
        solid::thermohyperelastic::Thermohyperelastic,
        thermal::conduction::ThermalConduction,
    },
    math::Quantity,
    mechanics::DeformationGradient,
    units::{EnergyDensity, Temperature},
};

impl<C1, C2> Thermohyperelastic for Canonical<C1, C2>
where
    C1: Thermohyperelastic,
    C2: ThermalConduction,
{
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        self.solid_constitutive_model()
            .helmholtz_free_energy_density(deformation_gradient, temperature)
    }
}
