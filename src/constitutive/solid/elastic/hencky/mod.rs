#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        Constitutive, ConstitutiveError,
        solid::{Solid, TWO_THIRDS, elastic::Elastic},
    },
    math::{ContractThirdFourthIndicesWithFirstSecondIndicesOf, IDENTITY_00, Rank2},
    mechanics::{
        Deformation, DeformationGradient, Scalar, SecondPiolaKirchhoffStress,
        SecondPiolaKirchhoffTangentStiffness,
    },
};

#[doc = include_str!("doc.md")]
#[derive(Debug)]
pub struct Hencky {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Scalar,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Scalar,
}

impl Solid for Hencky {
    fn bulk_modulus(&self) -> &Scalar {
        &self.bulk_modulus
    }
    fn shear_modulus(&self) -> &Scalar {
        &self.shear_modulus
    }
}

impl Elastic for Hencky {
    #[doc = include_str!("second_piola_kirchhoff_stress.md")]
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let (deviatoric_strain, strain_trace) =
            (deformation_gradient.right_cauchy_green().logm() * 0.5).deviatoric_and_trace();
        Ok(deviatoric_strain * (2.0 * self.shear_modulus())
            + IDENTITY_00 * (self.bulk_modulus() * strain_trace))
    }
    #[doc = include_str!("second_piola_kirchhoff_tangent_stiffness.md")]
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let right_cauchy_green = deformation_gradient.right_cauchy_green();
        let deformation_gradient_transpose = deformation_gradient.transpose();
        let scaled_deformation_gradient_transpose =
            &deformation_gradient_transpose * self.shear_modulus();
        Ok((right_cauchy_green
            .dlogm()
            .contract_third_fourth_indices_with_first_second_indices_of(
                &(SecondPiolaKirchhoffTangentStiffness::dyad_il_jk(
                    &IDENTITY_00,
                    &scaled_deformation_gradient_transpose,
                ) + SecondPiolaKirchhoffTangentStiffness::dyad_ik_jl(
                    &scaled_deformation_gradient_transpose,
                    &IDENTITY_00,
                )),
            ))
            + (SecondPiolaKirchhoffTangentStiffness::dyad_ij_kl(
                &(IDENTITY_00 * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())),
                &deformation_gradient_transpose.inverse(),
            )))
    }
}
