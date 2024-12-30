#[cfg(test)]
mod test;

use super::*;
use crate::math::special::{inverse_langevin, langevin_derivative};

/// The Arruda-Boyce hyperelastic constitutive model.[^cite]
///
/// [^cite]: E.M. Arruda and M.C. Boyce, [J. Mech. Phys. Solids **41**, 389 (1993)](https://doi.org/10.1016/0022-5096(93)90013-6).
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear modulus $`\mu`$.
/// - The number of links $`N_b`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The nondimensional end-to-end length per link of the chains is $`\gamma=\sqrt{\mathrm{tr}(\mathbf{B}^*)/3N_b}`$.
/// - The nondimensional force is given by the inverse Langevin function as $`\eta=\mathcal{L}^{-1}(\gamma)`$.
/// - The initial values are given by $`\gamma_0=\sqrt{1/3N_b}`$ and $`\eta_0=\mathcal{L}^{-1}(\gamma_0)`$.
/// - The Arruda-Boyce model reduces to the [Neo-Hookean model](NeoHookean) when $`N_b\to\infty`$.
#[derive(Debug)]
pub struct ArrudaBoyce<'a> {
    parameters: Parameters<'a>,
}

impl ArrudaBoyce<'_> {
    /// Returns the number of links.
    fn get_number_of_links(&self) -> &Scalar {
        &self.parameters[2]
    }
}

impl<'a> Constitutive<'a> for ArrudaBoyce<'a> {
    fn new(parameters: Parameters<'a>) -> Self {
        Self { parameters }
    }
}

impl<'a> Solid<'a> for ArrudaBoyce<'a> {
    fn get_bulk_modulus(&self) -> &Scalar {
        &self.parameters[0]
    }
    fn get_shear_modulus(&self) -> &Scalar {
        &self.parameters[1]
    }
}

impl<'a> Elastic<'a> for ArrudaBoyce<'a> {
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}) = \frac{\mu\gamma_0\eta}{J\gamma\eta_0}\,{\mathbf{B}^*}' + \frac{\kappa}{2}\left(J - \frac{1}{J}\right)\mathbf{1}
    /// ```
    fn calculate_cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = deformation_gradient.determinant();
        if jacobian > 0.0 {
            let (
                deviatoric_isochoric_left_cauchy_green_deformation,
                isochoric_left_cauchy_green_deformation_trace,
            ) = (self.calculate_left_cauchy_green_deformation(deformation_gradient)
                / jacobian.powf(TWO_THIRDS))
            .deviatoric_and_trace();
            let gamma =
                (isochoric_left_cauchy_green_deformation_trace / 3.0 / self.get_number_of_links())
                    .sqrt();
            if gamma >= 1.0 {
                Err(ConstitutiveError::Custom(
                    "Maximum extensibility reached.".to_string(),
                    deformation_gradient.copy(),
                    format!("{:?}", &self),
                ))
            } else {
                let gamma_0 = (1.0 / self.get_number_of_links()).sqrt();
                Ok(deviatoric_isochoric_left_cauchy_green_deformation
                    * (self.get_shear_modulus() * inverse_langevin(gamma)
                        / inverse_langevin(gamma_0)
                        * gamma_0
                        / gamma
                        / jacobian)
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
    /// \mathcal{T}_{ijkL}(\mathbf{F}) = \frac{\mu\gamma_0\eta}{J^{5/3}\gamma\eta_0}\left(\delta_{ik}F_{jL} + \delta_{jk}F_{iL} - \frac{2}{3}\,\delta_{ij}F_{kL}- \frac{5}{3} \, B_{ij}'F_{kL}^{-T} \right) + \frac{\mu\gamma_0\eta}{3J^{7/3}N_b\gamma^2\eta_0}\left(\frac{1}{\eta\mathcal{L}'(\eta)} - \frac{1}{\gamma}\right)B_{ij}'B_{km}'F_{mL}^{-T} + \frac{\kappa}{2} \left(J + \frac{1}{J}\right)\delta_{ij}F_{kL}^{-T}
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
            let deviatoric_left_cauchy_green_deformation =
                left_cauchy_green_deformation.deviatoric();
            let (
                deviatoric_isochoric_left_cauchy_green_deformation,
                isochoric_left_cauchy_green_deformation_trace,
            ) = (left_cauchy_green_deformation / jacobian.powf(TWO_THIRDS)).deviatoric_and_trace();
            let gamma =
                (isochoric_left_cauchy_green_deformation_trace / 3.0 / self.get_number_of_links())
                    .sqrt();
            if gamma >= 1.0 {
                Err(ConstitutiveError::Custom(
                    "Maximum extensibility reached.".to_string(),
                    deformation_gradient.copy(),
                    format!("{:?}", &self),
                ))
            } else {
                let gamma_0 = (1.0 / self.get_number_of_links()).sqrt();
                let eta = inverse_langevin(gamma);
                let scaled_shear_modulus =
                    gamma_0 / inverse_langevin(gamma_0) * self.get_shear_modulus() * eta
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
                            / self.get_number_of_links()
                            / gamma)),
                );
                Ok(
                    (CauchyTangentStiffness::dyad_ik_jl(&IDENTITY, deformation_gradient)
                        + CauchyTangentStiffness::dyad_il_jk(deformation_gradient, &IDENTITY)
                        - CauchyTangentStiffness::dyad_ij_kl(&IDENTITY, deformation_gradient)
                            * (TWO_THIRDS))
                        * scaled_shear_modulus
                        + CauchyTangentStiffness::dyad_ij_kl(
                            &(IDENTITY
                                * (0.5 * self.get_bulk_modulus() * (jacobian + 1.0 / jacobian))
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
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a(\mathbf{F}) = \frac{3\mu N_b\gamma_0}{\eta_0}\left[\gamma\eta - \gamma_0\eta_0 - \ln\left(\frac{\eta_0\sinh\eta}{\eta\sinh\eta_0}\right) \right] + \frac{\kappa}{2}\left[\frac{1}{2}\left(J^2 - 1\right) - \ln J\right]
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
            let gamma = (isochoric_left_cauchy_green_deformation.trace()
                / 3.0
                / self.get_number_of_links())
            .sqrt();
            if gamma >= 1.0 {
                Err(ConstitutiveError::Custom(
                    "Maximum extensibility reached.".to_string(),
                    deformation_gradient.copy(),
                    format!("{:?}", &self),
                ))
            } else {
                let eta = inverse_langevin(gamma);
                let gamma_0 = (1.0 / self.get_number_of_links()).sqrt();
                let eta_0 = inverse_langevin(gamma_0);
                Ok(3.0 * gamma_0 / eta_0
                    * self.get_shear_modulus()
                    * self.get_number_of_links()
                    * (gamma * eta
                        - gamma_0 * eta_0
                        - (eta_0 * eta.sinh() / (eta * eta_0.sinh())).ln())
                    + 0.5
                        * self.get_bulk_modulus()
                        * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln()))
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
