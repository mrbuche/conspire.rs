#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        hybrid::Multiplicative,
        solid::hyperelastic::Hyperelastic,
    },
    mechanics::{DeformationGradient, Scalar},
};

impl<C1, C2> Hyperelastic for Multiplicative<C1, C2>
where
    C1: Hyperelastic,
    C2: Hyperelastic,
{
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a(\mathbf{F}) = a_1(\mathbf{F}_1) + a_2(\mathbf{F}_2)
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        let (deformation_gradient_1, deformation_gradient_2) =
            self.deformation_gradients(deformation_gradient)?;
        Ok(self
            .0
            .helmholtz_free_energy_density(&deformation_gradient_1)?
            + self
                .1
                .helmholtz_free_energy_density(&deformation_gradient_2)?)
    }
}
