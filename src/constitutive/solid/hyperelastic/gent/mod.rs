#[cfg(test)]
mod test;

use super::*;

/// The Gent hyperelastic constitutive model.[^cite]
///
/// [^cite]: A.N. Gent, [Rubber Chem. Technol. **69**, 59 (1996)](https://doi.org/10.5254/1.3538357).
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear modulus $`\mu`$.
/// - The extensibility $`J_m`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Gent model reduces to the [Neo-Hookean model](NeoHookean) when $`J_m\to\infty`$.
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
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}) = \frac{J^{-1}\mu J_m {\mathbf{B}^* }'}{J_m - \mathrm{tr}(\mathbf{B}^* ) + 3} + \frac{\kappa}{2}\left(J - \frac{1}{J}\right)\mathbf{1}
    /// ```
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
    /// Calculates and returns the tangent stiffness associated with the Cauchy stress.
    ///
    /// ```math
    /// \mathcal{T}_{ijkL}(\mathbf{F}) = \frac{J^{-5/3}\mu J_m}{J_m - \mathrm{tr}(\mathbf{B}^* ) + 3}\Bigg[ \delta_{ik}F_{jL} + \delta_{jk}F_{iL} - \frac{2}{3}\,\delta_{ij}F_{kL} + \frac{2{B_{ij}^* }' F_{kL}}{J_m - \mathrm{tr}(\mathbf{B}^* ) + 3} - \left(\frac{5}{3} + \frac{2}{3}\frac{\mathrm{tr}(\mathbf{B}^* )}{J_m - \mathrm{tr}(\mathbf{B}^* ) + 3}\right) J^{2/3} {B_{ij}^* }' F_{kL}^{-T} \Bigg] + \frac{\kappa}{2} \left(J + \frac{1}{J}\right)\delta_{ij}F_{kL}^{-T}
    /// ```
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
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a(\mathbf{F}) = -\frac{\mu J_m}{2}\,\ln\left[1 - \frac{\mathrm{tr}(\mathbf{B}^* ) - 3}{J_m}\right] + \frac{\kappa}{2}\left[\frac{1}{2}\left(J^2 - 1\right) - \ln J\right]
    /// ```
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
