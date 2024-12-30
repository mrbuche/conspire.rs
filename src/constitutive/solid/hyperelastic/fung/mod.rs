#[cfg(test)]
mod test;

use super::*;

/// The Fung hyperelastic constitutive model.[^cite]
///
/// [^cite]: Y.C. Fung, [Am. J. Physiol. **213**, 1532 (1967)](https://doi.org/10.1152/ajplegacy.1967.213.6.1532).
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear modulus $`\mu`$.
/// - The extra modulus $`\mu_m`$.
/// - The exponent $`a`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Fung model reduces to the [Neo-Hookean model](NeoHookean) when $`\mu_m\to 0`$ or $`a\to 0`$.
#[derive(Debug)]
pub struct Fung<'a> {
    parameters: Parameters<'a>,
}

impl Fung<'_> {
    /// Returns the extra modulus.
    fn get_extra_modulus(&self) -> &Scalar {
        &self.parameters[2]
    }
    /// Returns the exponent.
    fn get_exponent(&self) -> &Scalar {
        &self.parameters[3]
    }
}

impl<'a> Constitutive<'a> for Fung<'a> {
    fn new(parameters: Parameters<'a>) -> Self {
        Self { parameters }
    }
}

impl<'a> Solid<'a> for Fung<'a> {
    fn get_bulk_modulus(&self) -> &Scalar {
        &self.parameters[0]
    }
    fn get_shear_modulus(&self) -> &Scalar {
        &self.parameters[1]
    }
}

impl<'a> Elastic<'a> for Fung<'a> {
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}) = \frac{1}{J}\left[\mu + \mu_m\left(e^{a[\mathrm{tr}(\mathbf{B}^* ) - 3]} - 1\right)\right]{\mathbf{B}^* }' + \frac{\kappa}{2}\left(J - \frac{1}{J}\right)\mathbf{1}
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
            Ok(deviatoric_isochoric_left_cauchy_green_deformation
                * ((self.get_shear_modulus()
                    + self.get_extra_modulus()
                        * ((self.get_exponent()
                            * (isochoric_left_cauchy_green_deformation_trace - 3.0))
                            .exp()
                            - 1.0))
                    / jacobian)
                + IDENTITY * self.get_bulk_modulus() * 0.5 * (jacobian - 1.0 / jacobian))
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
    /// \mathcal{T}_{ijkL}(\mathbf{F}) = \frac{1}{J^{5/3}}\left[\mu + \mu_m\left(e^{a[\mathrm{tr}(\mathbf{B}^* ) - 3]} - 1\right)\right]\left(\delta_{ik}F_{jL} + \delta_{jk}F_{iL} - \frac{2}{3}\,\delta_{ij}F_{kL} - \frac{5}{3} \, B_{ij}'F_{kL}^{-T} \right) + \frac{2a\mu_m}{J^{7/3}}\,e^{a[\mathrm{tr}(\mathbf{B}^* ) - 3]}B_{ij}'B_{km}'F_{mL}^{-T} + \frac{\kappa}{2} \left(J + \frac{1}{J}\right)\delta_{ij}F_{kL}^{-T}
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
            let exponential =
                (self.get_exponent() * (isochoric_left_cauchy_green_deformation_trace - 3.0)).exp();
            let scaled_shear_modulus_0 = (self.get_shear_modulus()
                + self.get_extra_modulus() * (exponential - 1.0))
                / jacobian.powf(FIVE_THIRDS);
            Ok(
                (CauchyTangentStiffness::dyad_ik_jl(&IDENTITY, deformation_gradient)
                    + CauchyTangentStiffness::dyad_il_jk(deformation_gradient, &IDENTITY)
                    - CauchyTangentStiffness::dyad_ij_kl(&IDENTITY, deformation_gradient)
                        * (TWO_THIRDS))
                    * scaled_shear_modulus_0
                    + CauchyTangentStiffness::dyad_ij_kl(
                        &deviatoric_isochoric_left_cauchy_green_deformation,
                        &((&deviatoric_isochoric_left_cauchy_green_deformation
                            * &inverse_transpose_deformation_gradient)
                            * (2.0 * self.get_exponent() * self.get_extra_modulus() * exponential
                                / jacobian)),
                    )
                    + CauchyTangentStiffness::dyad_ij_kl(
                        &(IDENTITY * (0.5 * self.get_bulk_modulus() * (jacobian + 1.0 / jacobian))
                            - self
                                .calculate_left_cauchy_green_deformation(deformation_gradient)
                                .deviatoric()
                                * (scaled_shear_modulus_0 * FIVE_THIRDS)),
                        &inverse_transpose_deformation_gradient,
                    ),
            )
        } else {
            Err(ConstitutiveError::InvalidJacobian(
                jacobian,
                deformation_gradient.copy(),
                format!("{:?}", &self),
            ))
        }
    }
}

impl<'a> Hyperelastic<'a> for Fung<'a> {
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a(\mathbf{F}) = \frac{\mu - \mu_m}{2}\left[\mathrm{tr}(\mathbf{B}^* ) - 3\right] + \frac{\mu_m}{2a}\left(e^{a[\mathrm{tr}(\mathbf{B}^* ) - 3]} - 1\right)
    /// ```
    fn calculate_helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        let jacobian = deformation_gradient.determinant();
        if jacobian > 0.0 {
            let scalar_term = self
                .calculate_left_cauchy_green_deformation(deformation_gradient)
                .trace()
                / jacobian.powf(TWO_THIRDS)
                - 3.0;
            Ok(0.5
                * ((self.get_shear_modulus() - self.get_extra_modulus()) * scalar_term
                    + self.get_extra_modulus() / self.get_exponent()
                        * ((self.get_exponent() * scalar_term).exp() - 1.0)
                    + self.get_bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln())))
        } else {
            Err(ConstitutiveError::InvalidJacobian(
                jacobian,
                deformation_gradient.copy(),
                format!("{:?}", &self),
            ))
        }
    }
}
