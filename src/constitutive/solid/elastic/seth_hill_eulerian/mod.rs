#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, TWO_THIRDS, elastic::Elastic},
    },
    math::{ContractThirdFourthWithFirstSecond, IDENTITY, Quantity, Rank2, Spectrum, TensorRank4},
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
    units::Stress,
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct SethHillEulerian {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The exponent $`m`$.
    pub exponent: Scalar,
}

impl SethHillEulerian {
    /// Returns the exponent.
    pub fn exponent(&self) -> Scalar {
        self.exponent
    }
}

impl Solid for SethHillEulerian {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Elastic for SethHillEulerian {
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let left_cauchy_green = deformation_gradient.left_cauchy_green();
        let (deviatoric_strain, strain_trace) = if self.exponent() == 0.0 {
            (left_cauchy_green
                .logm()
                .map_err(|error| ConstitutiveError::upstream(error, self))?
                * 0.5)
                .deviatoric_and_trace()
        } else {
            ((left_cauchy_green
                .powm(0.5 * self.exponent())
                .map_err(|error| ConstitutiveError::upstream(error, self))?
                - IDENTITY)
                / self.exponent())
            .deviatoric_and_trace()
        };
        Ok(deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
            + IDENTITY * (self.bulk_modulus() * strain_trace / jacobian))
    }
    #[doc = include_str!("cauchy_tangent_stiffness.md")]
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let left_cauchy_green = deformation_gradient.left_cauchy_green();
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        if self.exponent() == 0.0 {
            let (deviatoric_strain, strain_trace) = (left_cauchy_green
                .logm()
                .map_err(|error| ConstitutiveError::upstream(error, self))?
                * 0.5)
                .deviatoric_and_trace();
            let scaled_deformation_gradient =
                deformation_gradient * (self.shear_modulus() / jacobian);
            return Ok((left_cauchy_green
                .dlogm()
                .map_err(|error| ConstitutiveError::upstream(error, self))?
                .contract_third_fourth_with_first_second(
                    &(TensorRank4::dyad_il_jk(&scaled_deformation_gradient, &IDENTITY)
                        + TensorRank4::dyad_ik_jl(&IDENTITY, &scaled_deformation_gradient)),
                ))
                + TensorRank4::dyad_ij_kl(
                    &(IDENTITY
                        * ((self.bulk_modulus() - TWO_THIRDS * self.shear_modulus()) / jacobian)
                        - deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
                        - IDENTITY * (self.bulk_modulus() * strain_trace / jacobian)),
                    &inverse_transpose_deformation_gradient,
                ));
        }
        let half_exponent = 0.5 * self.exponent();
        let spectrum = Spectrum::new(&left_cauchy_green)
            .map_err(|error| ConstitutiveError::upstream(error, self))?;
        let (deviatoric_strain, strain_trace) = ((spectrum
            .powm(half_exponent)
            .map_err(|error| ConstitutiveError::upstream(error, self))?
            - IDENTITY)
            / self.exponent())
        .deviatoric_and_trace();
        let cauchy_stress = deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
            + IDENTITY * (self.bulk_modulus() * strain_trace / jacobian);
        let scaled_deformation_gradient =
            deformation_gradient * (2.0 * self.shear_modulus() / (self.exponent() * jacobian));
        let trace_term = spectrum
            .powm(half_exponent - 1.0)
            .map_err(|error| ConstitutiveError::upstream(error, self))?
            * deformation_gradient;
        Ok(spectrum
            .dpowm(half_exponent)
            .map_err(|error| ConstitutiveError::upstream(error, self))?
            .contract_third_fourth_with_first_second(
                &(TensorRank4::dyad_il_jk(&scaled_deformation_gradient, &IDENTITY)
                    + TensorRank4::dyad_ik_jl(&IDENTITY, &scaled_deformation_gradient)),
            )
            + TensorRank4::dyad_ij_kl(
                &(IDENTITY
                    * ((self.bulk_modulus() - TWO_THIRDS * self.shear_modulus()) / jacobian)),
                &trace_term,
            )
            - TensorRank4::dyad_ij_kl(&cauchy_stress, &inverse_transpose_deformation_gradient))
    }
}
