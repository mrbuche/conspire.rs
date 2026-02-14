use crate::{
    constitutive::{
        ConstitutiveError,
        hybrid::Additive,
        solid::{hyperelastic::Hyperelastic, hyperelastic_viscoplastic::HyperelasticViscoplastic},
    },
    math::Tensor,
    mechanics::{DeformationGradient, DeformationGradientPlastic, Scalar},
};

impl<C1, C2, Y> HyperelasticViscoplastic<Y> for Additive<C1, C2>
where
    C1: HyperelasticViscoplastic<Y>,
    C2: Hyperelastic,
    Y: Tensor,
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
