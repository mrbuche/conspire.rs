#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        hybrid::ElasticMultiplicative,
        solid::hyperelastic::{Hyperelastic, internal_variables::HyperelasticIV},
    },
    math::TensorRank4,
    mechanics::{
        DeformationGradient, DeformationGradient2, FirstPiolaKirchhoffTangentStiffness2, Scalar,
    },
};

impl<C1, C2>
    HyperelasticIV<
        DeformationGradient2,
        TensorRank4<3, 2, 0, 1, 0>,
        TensorRank4<3, 1, 0, 2, 0>,
        FirstPiolaKirchhoffTangentStiffness2,
    > for ElasticMultiplicative<C1, C2>
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
        deformation_gradient_2: &DeformationGradient2,
    ) -> Result<Scalar, ConstitutiveError> {
        let deformation_gradient_1 = deformation_gradient * deformation_gradient_2.inverse();
        Ok(self
            .0
            .helmholtz_free_energy_density(&deformation_gradient_1.into())?
            + self
                .1
                .helmholtz_free_energy_density(deformation_gradient_2.into())?)
    }
}
