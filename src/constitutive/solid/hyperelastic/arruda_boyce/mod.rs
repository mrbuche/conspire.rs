#[cfg(test)]
mod test;

use super::*;
use crate::math::special::{inverse_langevin, langevin_derivative};

#[doc = include_str!("model.md")]
#[derive(Debug)]
pub struct ArrudaBoyce<'a> {
    parameters: Parameters<'a>,
}

impl ArrudaBoyce<'_> {
    /// Returns the number of links.
    fn number_of_links(&self) -> &Scalar {
        &self.parameters[2]
    }
}

impl<'a> Constitutive<'a> for ArrudaBoyce<'a> {
    fn new(parameters: Parameters<'a>) -> Self {
        Self { parameters }
    }
}

impl<'a> Solid<'a> for ArrudaBoyce<'a> {
    fn bulk_modulus(&self) -> &Scalar {
        &self.parameters[0]
    }
    fn shear_modulus(&self) -> &Scalar {
        &self.parameters[1]
    }
}

impl<'a> Elastic<'a> for ArrudaBoyce<'a> {
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = deformation_gradient.determinant();
        if jacobian > 0.0 {
            let (
                deviatoric_isochoric_left_cauchy_green_deformation,
                isochoric_left_cauchy_green_deformation_trace,
            ) = (self.left_cauchy_green_deformation(deformation_gradient)
                / jacobian.powf(TWO_THIRDS))
            .deviatoric_and_trace();
            let gamma =
                (isochoric_left_cauchy_green_deformation_trace / 3.0 / self.number_of_links())
                    .sqrt();
            if gamma >= 1.0 {
                Err(ConstitutiveError::Custom(
                    "Maximum extensibility reached.".to_string(),
                    deformation_gradient.copy(),
                    format!("{:?}", &self),
                ))
            } else {
                let gamma_0 = (1.0 / self.number_of_links()).sqrt();
                Ok(deviatoric_isochoric_left_cauchy_green_deformation
                    * (self.shear_modulus() * inverse_langevin(gamma) / inverse_langevin(gamma_0)
                        * gamma_0
                        / gamma
                        / jacobian)
                    + IDENTITY * self.bulk_modulus() * 0.5 * (jacobian - 1.0 / jacobian))
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
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = deformation_gradient.determinant();
        if jacobian > 0.0 {
            let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
            let left_cauchy_green_deformation =
                self.left_cauchy_green_deformation(deformation_gradient);
            let deviatoric_left_cauchy_green_deformation =
                left_cauchy_green_deformation.deviatoric();
            let (
                deviatoric_isochoric_left_cauchy_green_deformation,
                isochoric_left_cauchy_green_deformation_trace,
            ) = (left_cauchy_green_deformation / jacobian.powf(TWO_THIRDS)).deviatoric_and_trace();
            let gamma =
                (isochoric_left_cauchy_green_deformation_trace / 3.0 / self.number_of_links())
                    .sqrt();
            if gamma >= 1.0 {
                Err(ConstitutiveError::Custom(
                    "Maximum extensibility reached.".to_string(),
                    deformation_gradient.copy(),
                    format!("{:?}", &self),
                ))
            } else {
                let gamma_0 = (1.0 / self.number_of_links()).sqrt();
                let eta = inverse_langevin(gamma);
                let scaled_shear_modulus =
                    gamma_0 / inverse_langevin(gamma_0) * self.shear_modulus() * eta
                        / gamma
                        / jacobian.powf(FIVE_THIRDS);
                let scaled_deviatoric_isochoric_left_cauchy_green_deformation =
                    deviatoric_left_cauchy_green_deformation * scaled_shear_modulus;
                let term = CauchyTangentStiffness::dyad_ij_kl(
                    &scaled_deviatoric_isochoric_left_cauchy_green_deformation,
                    &(deviatoric_isochoric_left_cauchy_green_deformation
                        * &inverse_transpose_deformation_gradient
                        * ((1.0 / eta / langevin_derivative(eta) - 1.0 / gamma)
                            / 3.0
                            / self.number_of_links()
                            / gamma)),
                );
                Ok(
                    (CauchyTangentStiffness::dyad_ik_jl(&IDENTITY, deformation_gradient)
                        + CauchyTangentStiffness::dyad_il_jk(deformation_gradient, &IDENTITY)
                        - CauchyTangentStiffness::dyad_ij_kl(&IDENTITY, deformation_gradient)
                            * (TWO_THIRDS))
                        * scaled_shear_modulus
                        + CauchyTangentStiffness::dyad_ij_kl(
                            &(IDENTITY * (0.5 * self.bulk_modulus() * (jacobian + 1.0 / jacobian))
                                - scaled_deviatoric_isochoric_left_cauchy_green_deformation
                                    * (FIVE_THIRDS)),
                            &inverse_transpose_deformation_gradient,
                        )
                        + term,
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

impl<'a> Hyperelastic<'a> for ArrudaBoyce<'a> {
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        let jacobian = deformation_gradient.determinant();
        if jacobian > 0.0 {
            let isochoric_left_cauchy_green_deformation = self
                .left_cauchy_green_deformation(deformation_gradient)
                / jacobian.powf(TWO_THIRDS);
            let gamma =
                (isochoric_left_cauchy_green_deformation.trace() / 3.0 / self.number_of_links())
                    .sqrt();
            if gamma >= 1.0 {
                Err(ConstitutiveError::Custom(
                    "Maximum extensibility reached.".to_string(),
                    deformation_gradient.copy(),
                    format!("{:?}", &self),
                ))
            } else {
                let eta = inverse_langevin(gamma);
                let gamma_0 = (1.0 / self.number_of_links()).sqrt();
                let eta_0 = inverse_langevin(gamma_0);
                Ok(3.0 * gamma_0 / eta_0
                    * self.shear_modulus()
                    * self.number_of_links()
                    * (gamma * eta
                        - gamma_0 * eta_0
                        - (eta_0 * eta.sinh() / (eta * eta_0.sinh())).ln())
                    + 0.5 * self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln()))
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
