#[cfg(test)]
mod test;

use super::*;

/// The Yeoh hyperelastic constitutive model.[^cite]
///
/// [^cite]: O.H. Yeoh, [Rubber Chem. Technol. **66**, 754 (1993)](https://doi.org/10.5254/1.3538343).
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear modulus $`\mu`$.
/// - The extra moduli $`\mu_n`$ for $`n=2\ldots N`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Yeoh model reduces to the [Neo-Hookean model](NeoHookean) when $`\mu_n\to 0`$ for $`n=2\ldots N`$.
#[derive(Debug)]
pub struct Yeoh<'a> {
    parameters: Parameters<'a>,
}

impl Yeoh<'_> {
    /// Returns an array of the moduli.
    pub fn get_moduli(&self) -> &[Scalar] {
        &self.parameters[1..]
    }
    /// Returns an array of the extra moduli.
    pub fn get_extra_moduli(&self) -> &[Scalar] {
        &self.parameters[2..]
    }
}

impl<'a> Constitutive<'a> for Yeoh<'a> {
    fn new(parameters: Parameters<'a>) -> Self {
        Self { parameters }
    }
}

impl<'a> Solid<'a> for Yeoh<'a> {
    fn get_bulk_modulus(&self) -> &Scalar {
        &self.parameters[0]
    }
    fn get_shear_modulus(&self) -> &Scalar {
        &self.parameters[1]
    }
}

impl<'a> Elastic<'a> for Yeoh<'a> {
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}) = \sum_{n=1}^N \frac{n\mu_n}{J}\left[\mathrm{tr}(\mathbf{B}^* ) - 3\right]^{n-1}\,{\mathbf{B}^*}' + \frac{\kappa}{2}\left(J - \frac{1}{J}\right)\mathbf{1}
    /// ```
    fn calculate_cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = deformation_gradient.determinant();
        if jacobian > 0.0 {
            let (deviatoric_left_cauchy_green_deformation, left_cauchy_green_deformation_trace) =
                self.calculate_left_cauchy_green_deformation(deformation_gradient)
                    .deviatoric_and_trace();
            let scalar_term = left_cauchy_green_deformation_trace / jacobian.powf(TWO_THIRDS) - 3.0;
            Ok(deviatoric_left_cauchy_green_deformation
                * self
                    .get_moduli()
                    .iter()
                    .enumerate()
                    .map(|(n, modulus)| {
                        ((n as Scalar) + 1.0) * modulus * scalar_term.powi(n as i32)
                    })
                    .sum::<Scalar>()
                / jacobian.powf(FIVE_THIRDS)
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
    /// \mathcal{T}_{ijkL}(\mathbf{F}) = \sum_{n=1}^N \frac{n\mu_n}{J^{5/3}}\left[\mathrm{tr}(\mathbf{B}^* ) - 3\right]^{n-1}\left(\delta_{ik}F_{jL} + \delta_{jk}F_{iL} - \frac{2}{3}\,\delta_{ij}F_{kL}- \frac{5}{3} \, B_{ij}'F_{kL}^{-T} \right) + \sum_{n=2}^N \frac{2n(n-1)\mu_n}{J^{7/3}}\left[\mathrm{tr}(\mathbf{B}^* ) - 3\right]^{n-2}B_{ij}'B_{km}'F_{mL}^{-T} + \frac{\kappa}{2} \left(J + \frac{1}{J}\right)\delta_{ij}F_{kL}^{-T}
    /// ```
    fn calculate_cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = deformation_gradient.determinant();
        if jacobian > 0.0 {
            let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
            let left_cauchy_green_deformation =
                self.calculate_left_cauchy_green_deformation(deformation_gradient);
            let scalar_term =
                left_cauchy_green_deformation.trace() / jacobian.powf(TWO_THIRDS) - 3.0;
            let scaled_modulus = self
                .get_moduli()
                .iter()
                .enumerate()
                .map(|(n, modulus)| ((n as Scalar) + 1.0) * modulus * scalar_term.powi(n as i32))
                .sum::<Scalar>()
                / jacobian.powf(FIVE_THIRDS);
            let deviatoric_left_cauchy_green_deformation =
                left_cauchy_green_deformation.deviatoric();
            let last_term = CauchyTangentStiffness::dyad_ij_kl(
                &deviatoric_left_cauchy_green_deformation,
                &((left_cauchy_green_deformation.deviatoric()
                    * &inverse_transpose_deformation_gradient)
                    * (2.0
                        * self
                            .get_extra_moduli()
                            .iter()
                            .enumerate()
                            .map(|(n, modulus)| {
                                ((n as Scalar) + 2.0)
                                    * ((n as Scalar) + 1.0)
                                    * modulus
                                    * scalar_term.powi(n as i32)
                            })
                            .sum::<Scalar>()
                        / jacobian.powf(SEVEN_THIRDS))),
            );
            Ok(
                (CauchyTangentStiffness::dyad_ik_jl(&IDENTITY, deformation_gradient)
                    + CauchyTangentStiffness::dyad_il_jk(deformation_gradient, &IDENTITY)
                    - CauchyTangentStiffness::dyad_ij_kl(&IDENTITY, deformation_gradient)
                        * (TWO_THIRDS))
                    * scaled_modulus
                    + CauchyTangentStiffness::dyad_ij_kl(
                        &(IDENTITY * (0.5 * self.get_bulk_modulus() * (jacobian + 1.0 / jacobian))
                            - deviatoric_left_cauchy_green_deformation
                                * (scaled_modulus * FIVE_THIRDS)),
                        &inverse_transpose_deformation_gradient,
                    )
                    + last_term,
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

impl<'a> Hyperelastic<'a> for Yeoh<'a> {
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a(\mathbf{F}) = \sum_{n=1}^N \frac{\mu_n}{2}\left[\mathrm{tr}(\mathbf{B}^* ) - 3\right]^n + \frac{\kappa}{2}\left[\frac{1}{2}\left(J^2 - 1\right) - \ln J\right]
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
                * (self
                    .get_moduli()
                    .iter()
                    .enumerate()
                    .map(|(n, modulus)| modulus * scalar_term.powi((n + 1) as i32))
                    .sum::<Scalar>()
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
