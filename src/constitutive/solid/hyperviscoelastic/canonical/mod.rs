#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        canonical::Canonical,
        fluid::hyperviscous::Hyperviscous,
        solid::{hyperelastic::Hyperelastic, hyperviscoelastic::Hyperviscoelastic},
    },
    math::Quantity,
    mechanics::DeformationGradient,
    units::EnergyDensity,
};

impl<C1, C2> Hyperviscoelastic for Canonical<C1, C2>
where
    C1: Hyperelastic,
    C2: Hyperviscous,
{
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        self.0.helmholtz_free_energy_density(deformation_gradient)
    }
}
