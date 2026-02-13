#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        hybrid::Additive,
        solid::{hyperelastic::Hyperelastic, hyperelastic_viscoplastic::HyperelasticViscoplastic},
    },
    mechanics::{DeformationGradient, DeformationGradientPlastic, Scalar},
};

impl<C1, C2> HyperelasticViscoplastic for Additive<C1, C2>
where
    C1: HyperelasticViscoplastic,
    C2: Hyperelastic,
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
    ) -> Result<Scalar, ConstitutiveError> {
        Ok(self
            .0
            .helmholtz_free_energy_density(deformation_gradient, deformation_gradient_p)?
            + self.1.helmholtz_free_energy_density(deformation_gradient)?)
    }
}
