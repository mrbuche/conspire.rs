#[cfg(test)]
mod test;

use super::hencky::Hencky;
use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, TWO_THIRDS, elastic::Elastic},
    },
    math::{
        ContractThirdFourthWithFirstSecond, IDENTITY_00, Quantity, Rank2, Spectrum, TensorRank4,
    },
    mechanics::{
        Deformation, DeformationGradient, Scalar, SecondPiolaKirchhoffStress,
        SecondPiolaKirchhoffTangentStiffness,
    },
    units::Stress,
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct SethHillLagrangian {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The exponent $`m`$.
    pub exponent: Scalar,
}

impl SethHillLagrangian {
    /// Returns the exponent.
    pub fn exponent(&self) -> Scalar {
        self.exponent
    }
    fn hencky(&self) -> Hencky {
        Hencky {
            bulk_modulus: self.bulk_modulus,
            shear_modulus: self.shear_modulus,
        }
    }
}

impl Solid for SethHillLagrangian {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Elastic for SethHillLagrangian {
    #[doc = include_str!("second_piola_kirchhoff_stress.md")]
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        if self.exponent() == 0.0 {
            return self
                .hencky()
                .second_piola_kirchhoff_stress(deformation_gradient);
        }
        let _jacobian = self.jacobian(deformation_gradient)?;
        let right_cauchy_green = deformation_gradient.right_cauchy_green();
        let (deviatoric_strain, strain_trace) =
            ((right_cauchy_green.powm(0.5 * self.exponent())? - IDENTITY_00) / self.exponent())
                .deviatoric_and_trace();
        Ok(deviatoric_strain * (2.0 * self.shear_modulus())
            + IDENTITY_00 * (self.bulk_modulus() * strain_trace))
    }
    #[doc = include_str!("second_piola_kirchhoff_tangent_stiffness.md")]
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        if self.exponent() == 0.0 {
            return self
                .hencky()
                .second_piola_kirchhoff_tangent_stiffness(deformation_gradient);
        }
        let _jacobian = self.jacobian(deformation_gradient)?;
        let right_cauchy_green = deformation_gradient.right_cauchy_green();
        let half_exponent = 0.5 * self.exponent();
        let spectrum = Spectrum::new(&right_cauchy_green)?;
        let deformation_gradient_transpose = deformation_gradient.transpose();
        let scaled_deformation_gradient_transpose =
            &deformation_gradient_transpose * (2.0 * self.shear_modulus() / self.exponent());
        let trace_term = deformation_gradient * spectrum.powm(half_exponent - 1.0)?;
        Ok(spectrum
            .dpowm(half_exponent)?
            .contract_third_fourth_with_first_second(
                &(TensorRank4::dyad_il_jk(&IDENTITY_00, &scaled_deformation_gradient_transpose)
                    + TensorRank4::dyad_ik_jl(
                        &scaled_deformation_gradient_transpose,
                        &IDENTITY_00,
                    )),
            )
            + TensorRank4::dyad_ij_kl(
                &(IDENTITY_00 * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())),
                &trace_term,
            ))
    }
}
