#[cfg(test)]
mod test;

use super::*;

/// The Mooney-Rivlin hyperelastic constitutive model.[^cite1]<sup>,</sup>[^cite2]
///
/// [^cite1]: M. Mooney, [J. Appl. Phys. **11**, 582 (1940)](https://doi.org/10.1063/1.1712836).
/// [^cite2]: R.S. Rivlin, [Philos. Trans. R. Soc. London, Ser. A **241**, 379 (1948)](https://doi.org/10.1098/rsta.1948.0024).
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear modulus $`\mu`$.
/// - The extra modulus $`\mu_m`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Mooney-Rivlin model reduces to the [Neo-Hookean model](NeoHookean) when $`\mu_m\to 0`$.
#[derive(Debug)]
pub struct MooneyRivlin<'a> {
    parameters: Parameters<'a>,
}

impl MooneyRivlin<'_> {
    /// Returns the extra modulus.
    fn get_extra_modulus(&self) -> &Scalar {
        &self.parameters[2]
    }
}

impl<'a> Constitutive<'a> for MooneyRivlin<'a> {
    fn new(parameters: Parameters<'a>) -> Self {
        Self { parameters }
    }
}

impl<'a> Solid<'a> for MooneyRivlin<'a> {
    fn get_bulk_modulus(&self) -> &Scalar {
        &self.parameters[0]
    }
    fn get_shear_modulus(&self) -> &Scalar {
        &self.parameters[1]
    }
}

impl<'a> Elastic<'a> for MooneyRivlin<'a> {
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}) = \frac{\mu - \mu_m}{J}\, {\mathbf{B}^* }' - \frac{\mu_m}{J}\left(\mathbf{B}^{* -1}\right)' + \frac{\kappa}{2}\left(J - \frac{1}{J}\right)\mathbf{1}
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
            Ok(((isochoric_left_cauchy_green_deformation.deviatoric()
                * (self.get_shear_modulus() - self.get_extra_modulus())
                - isochoric_left_cauchy_green_deformation
                    .inverse()
                    .deviatoric()
                    * self.get_extra_modulus())
                + IDENTITY * (self.get_bulk_modulus() * 0.5 * (jacobian.powi(2) - 1.0)))
                / jacobian)
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
    /// \mathcal{T}_{ijkL}(\mathbf{F}) = \frac{\mu-\mu_m}{J^{5/3}}\left(\delta_{ik}F_{jL} + \delta_{jk}F_{iL} - \frac{2}{3}\,\delta_{ij}F_{kL}- \frac{5}{3} \, B_{ij}'F_{kL}^{-T} \right) - \frac{\mu_m}{J}\left[ \frac{2}{3}\,B_{ij}^{* -1}F_{kL}^{-T} - B_{ik}^{* -1}F_{jL}^{-T} - B_{ik}^{* -1}F_{iL}^{-T} + \frac{2}{3}\,\delta_{ij}\left(B_{km}^{* -1}\right)'F_{mL}^{-T} - \left(B_{ij}^{* -1}\right)'F_{kL}^{-T} \right] + \frac{\kappa}{2} \left(J + \frac{1}{J}\right)\delta_{ij}F_{kL}^{-T}
    /// ```
    fn calculate_cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = deformation_gradient.determinant();
        if jacobian > 0.0 {
            let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
            let scaled_delta_shear_modulus =
                (self.get_shear_modulus() - self.get_extra_modulus()) / jacobian.powf(FIVE_THIRDS);
            let inverse_isochoric_left_cauchy_green_deformation = (self
                .calculate_left_cauchy_green_deformation(deformation_gradient)
                / jacobian.powf(TWO_THIRDS))
            .inverse();
            let deviatoric_inverse_isochoric_left_cauchy_green_deformation =
                inverse_isochoric_left_cauchy_green_deformation.deviatoric();
            let term_1 = CauchyTangentStiffness::dyad_ij_kl(
                &inverse_isochoric_left_cauchy_green_deformation,
                &inverse_transpose_deformation_gradient,
            ) * TWO_THIRDS
                - CauchyTangentStiffness::dyad_ik_jl(
                    &inverse_isochoric_left_cauchy_green_deformation,
                    &inverse_transpose_deformation_gradient,
                )
                - CauchyTangentStiffness::dyad_il_jk(
                    &inverse_transpose_deformation_gradient,
                    &inverse_isochoric_left_cauchy_green_deformation,
                );
            let term_3 = CauchyTangentStiffness::dyad_ij_kl(
                &deviatoric_inverse_isochoric_left_cauchy_green_deformation,
                &inverse_transpose_deformation_gradient,
            );
            let term_2 = CauchyTangentStiffness::dyad_ij_kl(
                &IDENTITY,
                &((deviatoric_inverse_isochoric_left_cauchy_green_deformation * TWO_THIRDS)
                    * &inverse_transpose_deformation_gradient),
            );
            Ok(
                (CauchyTangentStiffness::dyad_ik_jl(&IDENTITY, deformation_gradient)
                    + CauchyTangentStiffness::dyad_il_jk(deformation_gradient, &IDENTITY)
                    - CauchyTangentStiffness::dyad_ij_kl(&IDENTITY, deformation_gradient)
                        * (TWO_THIRDS))
                    * scaled_delta_shear_modulus
                    + CauchyTangentStiffness::dyad_ij_kl(
                        &(IDENTITY * (0.5 * self.get_bulk_modulus() * (jacobian + 1.0 / jacobian))
                            - self
                                .calculate_left_cauchy_green_deformation(deformation_gradient)
                                .deviatoric()
                                * (scaled_delta_shear_modulus * FIVE_THIRDS)),
                        &inverse_transpose_deformation_gradient,
                    )
                    - (term_1 + term_2 - term_3) * self.get_extra_modulus() / jacobian,
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

impl<'a> Hyperelastic<'a> for MooneyRivlin<'a> {
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a(\mathbf{F}) = \frac{\mu - \mu_m}{2}\left[\mathrm{tr}(\mathbf{B}^* ) - 3\right] + \frac{\mu_m}{2}\left[I_2(\mathbf{B}^*) - 3\right] + \frac{\kappa}{2}\left[\frac{1}{2}\left(J^2 - 1\right) - \ln J\right]
    /// ```
    fn calculate_helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        let jacobian = deformation_gradient.determinant();
        if jacobian > 0.0 {
            let isochoric_left_cauchy_green_deformation = self
                .calculate_left_cauchy_green_deformation(deformation_gradient)
                / jacobian.powf(TWO_THIRDS);
            Ok(0.5
                * ((self.get_shear_modulus() - self.get_extra_modulus())
                    * (isochoric_left_cauchy_green_deformation.trace() - 3.0)
                    + self.get_extra_modulus()
                        * (isochoric_left_cauchy_green_deformation.second_invariant() - 3.0)
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
