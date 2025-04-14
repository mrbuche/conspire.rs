#[cfg(test)]
mod test;

use super::*;

/// The Almansi-Hamel thermoelastic constitutive model.
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear modulus $`\mu`$.
/// - The coefficient of thermal expansion $`\alpha`$.
/// - The reference temperature $`T_\mathrm{ref}`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
/// - The temperature $`T`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Almansi-Hamel strain measure is given by $`\mathbf{e}=\tfrac{1}{2}(\mathbf{1}-\mathbf{B}^{-1})`$.
#[derive(Debug)]
pub struct AlmansiHamel<P> {
    parameters: P,
}

impl<P> Constitutive<P> for AlmansiHamel<P>
where
    P: Parameters,
{
    fn new(parameters: P) -> Self {
        Self { parameters }
    }
}

impl<P> Solid<P> for AlmansiHamel<P>
where
    P: Parameters,
{
    fn bulk_modulus(&self) -> &Scalar {
        self.parameters.get(0)
    }
    fn shear_modulus(&self) -> &Scalar {
        self.parameters.get(1)
    }
}

impl<P> Thermoelastic<P> for AlmansiHamel<P>
where
    P: Parameters,
{
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}, T) = \frac{2\mu}{J}\,\mathbf{e}' + \frac{\kappa}{J}\,\mathrm{tr}(\mathbf{e})\mathbf{1} - \frac{3\alpha\kappa}{J}(T - T_\mathrm{ref})\mathbf{1}
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_deformation_gradient = deformation_gradient.inverse();
        let strain = (IDENTITY
            - inverse_deformation_gradient.transpose() * &inverse_deformation_gradient)
            * 0.5;
        let (deviatoric_strain, strain_trace) = strain.deviatoric_and_trace();
        Ok(deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
            + IDENTITY
                * (self.bulk_modulus() / jacobian
                    * (strain_trace
                        - 3.0
                            * self.coefficient_of_thermal_expansion()
                            * (temperature - self.reference_temperature()))))
    }
    /// Calculates and returns the tangent stiffness associated with the Cauchy stress.
    ///
    /// ```math
    /// \mathcal{T}_{ijkL}(\mathbf{F}, T) = \frac{\mu}{J}\left[B_{jk}^{-1}F_{iL}^{-T} + B_{ik}^{-1}F_{jL}^{-T} - \frac{2}{3}\,\delta_{ij}B_{km}^{-1}F_{mL}^{-T} - 2e_{ij}'F_{kL}^{-T}\right] + \frac{\kappa}{J}\left\{\delta_{ij}B_{km}^{-1}F_{mL}^{-T} - \Big[\mathrm{tr}(\mathbf{e}) - 3\alpha(T - T_\mathrm{ref})\Big]\delta_{ij}F_{kL}^{-T}\right\}
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: &Scalar,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let inverse_left_cauchy_green_deformation = &inverse_transpose_deformation_gradient
            * inverse_transpose_deformation_gradient.transpose();
        let strain = (IDENTITY - &inverse_left_cauchy_green_deformation) * 0.5;
        let (deviatoric_strain, strain_trace) = strain.deviatoric_and_trace();
        Ok((CauchyTangentStiffness::dyad_il_jk(
            &inverse_transpose_deformation_gradient,
            &inverse_left_cauchy_green_deformation,
        ) + CauchyTangentStiffness::dyad_ik_jl(
            &inverse_left_cauchy_green_deformation,
            &inverse_transpose_deformation_gradient,
        )) * (self.shear_modulus() / jacobian)
            + CauchyTangentStiffness::dyad_ij_kl(
                &IDENTITY,
                &(inverse_left_cauchy_green_deformation
                    * &inverse_transpose_deformation_gradient
                    * ((self.bulk_modulus() - self.shear_modulus() * TWO_THIRDS) / jacobian)),
            )
            - CauchyTangentStiffness::dyad_ij_kl(
                &(deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
                    + IDENTITY
                        * (self.bulk_modulus() / jacobian
                            * (strain_trace
                                - 3.0
                                    * self.coefficient_of_thermal_expansion()
                                    * (temperature - self.reference_temperature())))),
                &inverse_transpose_deformation_gradient,
            ))
    }
    fn coefficient_of_thermal_expansion(&self) -> &Scalar {
        self.parameters.get(2)
    }
    fn reference_temperature(&self) -> &Scalar {
        self.parameters.get(3)
    }
}
