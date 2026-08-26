#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, TWO_THIRDS, elastic::Elastic},
    },
    math::{IDENTITY_00, Quantity, Rank2, TensorRank4},
    mechanics::{
        DeformationGradient, SecondPiolaKirchhoffStress, SecondPiolaKirchhoffTangentStiffness,
    },
    units::Stress,
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct AlmansiHamelLagrangian {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
}

impl Solid for AlmansiHamelLagrangian {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Elastic for AlmansiHamelLagrangian {
    #[doc = include_str!("second_piola_kirchhoff_stress.md")]
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let inverse_deformation_gradient = deformation_gradient.inverse();
        let strain = (IDENTITY_00
            - &inverse_deformation_gradient * inverse_deformation_gradient.transpose())
            * 0.5;
        let (deviatoric_strain, strain_trace) = strain.deviatoric_and_trace();
        Ok(deviatoric_strain * (2.0 * self.shear_modulus())
            + IDENTITY_00 * (self.bulk_modulus() * strain_trace))
    }
    #[doc = include_str!("second_piola_kirchhoff_tangent_stiffness.md")]
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let inverse_deformation_gradient = deformation_gradient.inverse();
        let inverse_right_cauchy_green_deformation =
            &inverse_deformation_gradient * inverse_deformation_gradient.transpose();
        let scaled_inverse_deformation_gradient =
            &inverse_deformation_gradient * self.shear_modulus();
        Ok((TensorRank4::dyad_ik_jl(
            &scaled_inverse_deformation_gradient,
            &inverse_right_cauchy_green_deformation,
        ) + TensorRank4::dyad_il_jk(
            &inverse_right_cauchy_green_deformation,
            &scaled_inverse_deformation_gradient,
        )) + TensorRank4::dyad_ij_kl(
            &IDENTITY_00,
            &(inverse_deformation_gradient.transpose()
                * &inverse_right_cauchy_green_deformation
                * (self.bulk_modulus() - self.shear_modulus() * TWO_THIRDS)),
        ))
    }
}
