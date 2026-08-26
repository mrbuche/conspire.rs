#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, TWO_THIRDS, elastic::Elastic},
    },
    math::{IDENTITY, Quantity, Rank2, TensorRank4},
    mechanics::{CauchyStress, CauchyTangentStiffness, DeformationGradient},
    units::Stress,
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct AlmansiHamelEulerian {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
}

impl Solid for AlmansiHamelEulerian {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Elastic for AlmansiHamelEulerian {
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_deformation_gradient = deformation_gradient.inverse();
        let strain = (IDENTITY
            - inverse_deformation_gradient.transpose() * &inverse_deformation_gradient)
            * 0.5;
        let (deviatoric_strain, strain_trace) = strain.deviatoric_and_trace();
        Ok(deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
            + IDENTITY * (self.bulk_modulus() * strain_trace / jacobian))
    }
    #[doc = include_str!("cauchy_tangent_stiffness.md")]
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let inverse_left_cauchy_green_deformation = &inverse_transpose_deformation_gradient
            * inverse_transpose_deformation_gradient.transpose();
        let scaled_inverse_left_cauchy_green_deformation =
            &inverse_left_cauchy_green_deformation * (self.shear_modulus() / jacobian);
        let strain = (IDENTITY - &inverse_left_cauchy_green_deformation) * 0.5;
        let (deviatoric_strain, strain_trace) = strain.deviatoric_and_trace();
        Ok((TensorRank4::dyad_il_jk(
            &inverse_transpose_deformation_gradient,
            &scaled_inverse_left_cauchy_green_deformation,
        ) + TensorRank4::dyad_ik_jl(
            &scaled_inverse_left_cauchy_green_deformation,
            &inverse_transpose_deformation_gradient,
        )) + TensorRank4::dyad_ij_kl(
            &IDENTITY,
            &(inverse_left_cauchy_green_deformation
                * &inverse_transpose_deformation_gradient
                * ((self.bulk_modulus() - self.shear_modulus() * TWO_THIRDS) / jacobian)),
        ) - TensorRank4::dyad_ij_kl(
            &(deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
                + IDENTITY * (self.bulk_modulus() * strain_trace / jacobian)),
            &inverse_transpose_deformation_gradient,
        ))
    }
}
