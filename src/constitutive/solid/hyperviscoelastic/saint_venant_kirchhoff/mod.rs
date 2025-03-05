#[cfg(test)]
mod test;

use super::*;

/// The Saint Venant-Kirchhoff hyperviscoelastic constitutive model.
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear modulus $`\mu`$.
/// - The bulk viscosity $`\zeta`$.
/// - The shear viscosity $`\eta`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
/// - The deformation gradient rate $`\dot{\mathbf{F}}`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Green-Saint Venant strain measure is given by $`\mathbf{E}=\tfrac{1}{2}(\mathbf{C}-\mathbf{1})`$.
#[derive(Debug)]
pub struct SaintVenantKirchhoff<'a> {
    parameters: Parameters<'a>,
}

impl<'a> Constitutive<'a> for SaintVenantKirchhoff<'a> {
    fn new(parameters: Parameters<'a>) -> Self {
        Self { parameters }
    }
}

impl<'a> Solid<'a> for SaintVenantKirchhoff<'a> {
    fn bulk_modulus(&self) -> &Scalar {
        &self.parameters[0]
    }
    fn shear_modulus(&self) -> &Scalar {
        &self.parameters[1]
    }
}

impl<'a> Viscous<'a> for SaintVenantKirchhoff<'a> {
    fn bulk_viscosity(&self) -> &Scalar {
        &self.parameters[2]
    }
    fn shear_viscosity(&self) -> &Scalar {
        &self.parameters[3]
    }
}

impl<'a> Viscoelastic<'a> for SaintVenantKirchhoff<'a> {
    /// Calculates and returns the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{S}(\mathbf{F},\dot\mathbf{F}) = 2\mu\mathbf{E}' + \kappa\,\mathrm{tr}(\mathbf{E})\mathbf{1} + 2\eta\dot{\mathbf{E}}' + \zeta\,\mathrm{tr}(\dot{\mathbf{E}})\mathbf{1}
    /// ```
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        let jacobian = deformation_gradient.determinant();
        if jacobian > 0.0 {
            let (deviatoric_strain, strain_trace) =
                ((self.right_cauchy_green_deformation(deformation_gradient) - IDENTITY_00) * 0.5)
                    .deviatoric_and_trace();
            let first_term = deformation_gradient_rate.transpose() * deformation_gradient;
            let (deviatoric_strain_rate, strain_rate_trace) =
                ((&first_term + first_term.transpose()) * 0.5).deviatoric_and_trace();
            Ok(deviatoric_strain * (2.0 * self.shear_modulus())
                + deviatoric_strain_rate * (2.0 * self.shear_viscosity())
                + IDENTITY_00
                    * (self.bulk_modulus() * strain_trace
                        + self.bulk_viscosity() * strain_rate_trace))
        } else {
            Err(ConstitutiveError::InvalidJacobian(
                jacobian,
                deformation_gradient.clone(),
                format!("{:?}", &self),
            ))
        }
    }
    /// Calculates and returns the rate tangent stiffness associated with the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{W}_{IJkL}(\mathbf{F}) = \eta\,\delta_{JL}F_{kI} + \eta\,\delta_{IL}F_{kJ} + \left(\zeta - \frac{2}{3}\,\eta\right)\delta_{IJ}F_{kL}
    /// ```
    fn second_piola_kirchhoff_rate_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        _: &DeformationGradientRate,
    ) -> Result<SecondPiolaKirchhoffRateTangentStiffness, ConstitutiveError> {
        let jacobian = deformation_gradient.determinant();
        if jacobian > 0.0 {
            let scaled_deformation_gradient_transpose =
                deformation_gradient.transpose() * self.shear_viscosity();
            Ok(SecondPiolaKirchhoffRateTangentStiffness::dyad_ik_jl(
                &scaled_deformation_gradient_transpose,
                &IDENTITY_00,
            ) + SecondPiolaKirchhoffRateTangentStiffness::dyad_il_jk(
                &IDENTITY_00,
                &scaled_deformation_gradient_transpose,
            ) + SecondPiolaKirchhoffRateTangentStiffness::dyad_ij_kl(
                &(IDENTITY_00 * (self.bulk_viscosity() - TWO_THIRDS * self.shear_viscosity())),
                deformation_gradient,
            ))
        } else {
            Err(ConstitutiveError::InvalidJacobian(
                jacobian,
                deformation_gradient.clone(),
                format!("{:?}", &self),
            ))
        }
    }
}

impl<'a> ElasticHyperviscous<'a> for SaintVenantKirchhoff<'a> {
    /// Calculates and returns the viscous dissipation.
    ///
    /// ```math
    /// \phi(\mathbf{F},\dot{\mathbf{F}}) = \eta\,\mathrm{tr}(\dot{\mathbf{E}}^2) + \frac{1}{2}\left(\zeta - \frac{2}{3}\,\eta\right)\mathrm{tr}(\dot{\mathbf{E}})^2
    /// ```
    fn viscous_dissipation(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<Scalar, ConstitutiveError> {
        let jacobian = deformation_gradient.determinant();
        if jacobian > 0.0 {
            let first_term = deformation_gradient_rate.transpose() * deformation_gradient;
            let strain_rate = (&first_term + first_term.transpose()) * 0.5;
            Ok(self.shear_viscosity() * strain_rate.squared_trace()
                + 0.5
                    * (self.bulk_viscosity() - TWO_THIRDS * self.shear_viscosity())
                    * strain_rate.trace().powi(2))
        } else {
            Err(ConstitutiveError::InvalidJacobian(
                jacobian,
                deformation_gradient.clone(),
                format!("{:?}", &self),
            ))
        }
    }
}

impl<'a> Hyperviscoelastic<'a> for SaintVenantKirchhoff<'a> {
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a(\mathbf{F}) = \mu\,\mathrm{tr}(\mathbf{E}^2) + \frac{1}{2}\left(\kappa - \frac{2}{3}\,\mu\right)\mathrm{tr}(\mathbf{E})^2
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        let jacobian = deformation_gradient.determinant();
        if jacobian > 0.0 {
            let strain =
                (self.right_cauchy_green_deformation(deformation_gradient) - IDENTITY_00) * 0.5;
            Ok(self.shear_modulus() * strain.squared_trace()
                + 0.5
                    * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())
                    * strain.trace().powi(2))
        } else {
            Err(ConstitutiveError::InvalidJacobian(
                jacobian,
                deformation_gradient.clone(),
                format!("{:?}", &self),
            ))
        }
    }
}
