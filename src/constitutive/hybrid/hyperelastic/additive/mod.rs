#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        hybrid::{Additive, Hybrid},
        solid::hyperelastic::Hyperelastic,
        ConstitutiveError,
    },
    mechanics::{DeformationGradient, Scalar},
};

impl<'a, C1: Hyperelastic<'a>, C2: Hyperelastic<'a>> Hyperelastic<'a> for Additive<C1, C2> {
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a(\mathbf{F}) = a_1(\mathbf{F}) + a_2(\mathbf{F})
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        Ok(self
            .constitutive_model_1()
            .helmholtz_free_energy_density(deformation_gradient)?
            + self
                .constitutive_model_2()
                .helmholtz_free_energy_density(deformation_gradient)?)
    }
}
