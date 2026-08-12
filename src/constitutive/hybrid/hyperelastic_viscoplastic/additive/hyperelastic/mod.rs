use crate::math::Quantity;
use crate::math::unit::EnergyDensity;
use crate::{
    constitutive::{
        ConstitutiveError,
        hybrid::ElasticViscoplasticAdditiveElastic,
        solid::{hyperelastic::Hyperelastic, hyperelastic_viscoplastic::HyperelasticViscoplastic},
    },
    math::{Differentiate, Tensor},
    mechanics::{DeformationGradient, DeformationGradientPlastic},
};

impl<C1, C2, Y1> HyperelasticViscoplastic<Y1> for ElasticViscoplasticAdditiveElastic<C1, C2, Y1>
where
    C1: HyperelasticViscoplastic<Y1>,
    C2: Hyperelastic,
    Y1: Differentiate + Tensor,
{
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a = a_1(\mathbf{F},\mathbf{F}_\mathrm{p}) + a_2(\mathbf{F})
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        Ok(self
            .0
            .helmholtz_free_energy_density(deformation_gradient, deformation_gradient_p)?
            + self.1.helmholtz_free_energy_density(deformation_gradient)?)
    }
}
