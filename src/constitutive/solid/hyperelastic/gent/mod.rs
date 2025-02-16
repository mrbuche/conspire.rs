#[cfg(test)]
mod test;

use super::*;

#[doc = include_str!("model.md")]
#[derive(Debug)]
pub struct Gent<'a> {
    parameters: Parameters<'a>,
}

impl Gent<'_> {
    /// Returns the extensibility.
    fn get_extensibility(&self) -> &Scalar {
        &self.parameters[2]
    }
}

impl<'a> Constitutive<'a> for Gent<'a> {
    fn new(parameters: Parameters<'a>) -> Self {
        Self { parameters }
    }
}

impl<'a> Solid<'a> for Gent<'a> {
    fn get_bulk_modulus(&self) -> &Scalar {
        &self.parameters[0]
    }
    fn get_shear_modulus(&self) -> &Scalar {
        &self.parameters[1]
    }
}

impl<'a> Elastic<'a> for Gent<'a> {
    #[doc = include_str!("cauchy_stress.md")]
    fn calculate_cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = deformation_gradient.determinant();
        if jacobian > 0.0 {
            let isochoric_left_cauchy_green_deformation = self
                .calculate_left_cauchy_green_deformation(deformation_gradient)
                / jacobian.powf(TWO_THIRDS);
            let (
                deviatoric_isochoric_left_cauchy_green_deformation,
                isochoric_left_cauchy_green_deformation_trace,
            ) = isochoric_left_cauchy_green_deformation.deviatoric_and_trace();
            let denominator =
                self.get_extensibility() - isochoric_left_cauchy_green_deformation_trace + 3.0;
            if denominator <= 0.0 {
                Err(ConstitutiveError::Custom(
                    "Maximum extensibility reached.".to_string(),
                    deformation_gradient.copy(),
                    format!("{:?}", &self),
                ))
            } else {
                Ok((deviatoric_isochoric_left_cauchy_green_deformation
                    * self.get_shear_modulus()
                    * self.get_extensibility()
                    / jacobian)
                    / denominator
                    + IDENTITY * self.get_bulk_modulus() * 0.5 * (jacobian - 1.0 / jacobian))
            }
        } else {
            Err(ConstitutiveError::InvalidJacobian(
                jacobian,
                deformation_gradient.copy(),
                format!("{:?}", &self),
            ))
        }
    }
    #[doc = include_str!("cauchy_tangent_stiffness.md")]
    fn calculate_cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = deformation_gradient.determinant();
        if jacobian > 0.0 {
            let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
            let isochoric_left_cauchy_green_deformation = self
                .calculate_left_cauchy_green_deformation(deformation_gradient)
                / jacobian.powf(TWO_THIRDS);
            let (
                deviatoric_isochoric_left_cauchy_green_deformation,
                isochoric_left_cauchy_green_deformation_trace,
            ) = isochoric_left_cauchy_green_deformation.deviatoric_and_trace();
            let denominator =
                self.get_extensibility() - isochoric_left_cauchy_green_deformation_trace + 3.0;
            if denominator <= 0.0 {
                Err(ConstitutiveError::Custom(
                    "Maximum extensibility reached.".to_string(),
                    deformation_gradient.copy(),
                    format!("{:?}", &self),
                ))
            } else {
                let prefactor =
                    self.get_shear_modulus() * self.get_extensibility() / jacobian / denominator;
                Ok(
                    (CauchyTangentStiffness::dyad_ik_jl(&IDENTITY, deformation_gradient)
                        + CauchyTangentStiffness::dyad_il_jk(deformation_gradient, &IDENTITY)
                        - CauchyTangentStiffness::dyad_ij_kl(&IDENTITY, deformation_gradient)
                            * (TWO_THIRDS)
                        + CauchyTangentStiffness::dyad_ij_kl(
                            &deviatoric_isochoric_left_cauchy_green_deformation,
                            deformation_gradient,
                        ) * (2.0 / denominator))
                        * (prefactor / jacobian.powf(TWO_THIRDS))
                        + CauchyTangentStiffness::dyad_ij_kl(
                            &(IDENTITY
                                * (0.5 * self.get_bulk_modulus() * (jacobian + 1.0 / jacobian))
                                - deviatoric_isochoric_left_cauchy_green_deformation
                                    * prefactor
                                    * ((5.0
                                        + 2.0 * isochoric_left_cauchy_green_deformation_trace
                                            / denominator)
                                        / 3.0)),
                            &inverse_transpose_deformation_gradient,
                        ),
                )
            }
        } else {
            Err(ConstitutiveError::InvalidJacobian(
                jacobian,
                deformation_gradient.copy(),
                format!("{:?}", &self),
            ))
        }
    }
}

impl<'a> Hyperelastic<'a> for Gent<'a> {
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn calculate_helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        let jacobian = deformation_gradient.determinant();
        if jacobian > 0.0 {
            let factor = (self
                .calculate_left_cauchy_green_deformation(deformation_gradient)
                .trace()
                / jacobian.powf(TWO_THIRDS)
                - 3.0)
                / self.get_extensibility();
            if factor >= 1.0 {
                Err(ConstitutiveError::Custom(
                    "Maximum extensibility reached.".to_string(),
                    deformation_gradient.copy(),
                    format!("{:?}", &self),
                ))
            } else {
                Ok(0.5
                    * (-self.get_shear_modulus() * self.get_extensibility() * (1.0 - factor).ln()
                        + self.get_bulk_modulus()
                            * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln())))
            }
        } else {
            Err(ConstitutiveError::InvalidJacobian(
                jacobian,
                deformation_gradient.copy(),
                format!("{:?}", &self),
            ))
        }
    }
}
