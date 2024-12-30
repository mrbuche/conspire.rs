#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        hybrid::{Hybrid, Multiplicative, MultiplicativeTrait},
        solid::hyperelastic::Hyperelastic,
        ConstitutiveError,
    },
    mechanics::{DeformationGradient, Scalar},
};

impl<'a, C1: Hyperelastic<'a>, C2: Hyperelastic<'a>> Hyperelastic<'a> for Multiplicative<C1, C2> {
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a(\mathbf{F}) = a_1(\mathbf{F}_1) + a_2(\mathbf{F}_2)
    /// ```
    fn calculate_helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        let (deformation_gradient_1, deformation_gradient_2) =
            self.calculate_deformation_gradients(deformation_gradient)?;
        Ok(self
            .get_constitutive_model_1()
            .calculate_helmholtz_free_energy_density(&deformation_gradient_1)?
            + self
                .get_constitutive_model_2()
                .calculate_helmholtz_free_energy_density(&deformation_gradient_2)?)
    }
}
