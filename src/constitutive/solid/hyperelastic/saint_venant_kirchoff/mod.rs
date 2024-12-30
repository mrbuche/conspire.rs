#[cfg(test)]
mod test;

use super::*;

/// The Saint Venant-Kirchoff hyperelastic constitutive model.
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear modulus $`\mu`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Green-Saint Venant strain measure is given by $`\mathbf{E}=\tfrac{1}{2}(\mathbf{C}-\mathbf{1})`$.
#[derive(Debug)]
pub struct SaintVenantKirchoff<'a> {
    parameters: Parameters<'a>,
}

impl<'a> Constitutive<'a> for SaintVenantKirchoff<'a> {
    fn new(parameters: Parameters<'a>) -> Self {
        Self { parameters }
    }
}

impl<'a> Solid<'a> for SaintVenantKirchoff<'a> {
    fn get_bulk_modulus(&self) -> &Scalar {
        &self.parameters[0]
    }
    fn get_shear_modulus(&self) -> &Scalar {
        &self.parameters[1]
    }
}

impl<'a> Elastic<'a> for SaintVenantKirchoff<'a> {
    /// Calculates and returns the second Piola-Kirchoff stress.
    ///
    /// ```math
    /// \mathbf{S}(\mathbf{F}) = 2\mu\mathbf{E}' + \kappa\,\mathrm{tr}(\mathbf{E})\mathbf{1}
    /// ```
    fn calculate_second_piola_kirchoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchoffStress, ConstitutiveError> {
        let jacobian = deformation_gradient.determinant();
        if jacobian > 0.0 {
            let (deviatoric_strain, strain_trace) = ((self
                .calculate_right_cauchy_green_deformation(deformation_gradient)
                - IDENTITY_00)
                * 0.5)
                .deviatoric_and_trace();
            Ok(deviatoric_strain * (2.0 * self.get_shear_modulus())
                + IDENTITY_00 * (self.get_bulk_modulus() * strain_trace))
        } else {
            Err(ConstitutiveError::InvalidJacobian(
                jacobian,
                deformation_gradient.copy(),
                format!("{:?}", &self),
            ))
        }
    }
    /// Calculates and returns the tangent stiffness associated with the second Piola-Kirchoff stress.
    ///
    /// ```math
    /// \mathcal{G}_{IJkL}(\mathbf{F}) = \mu\,\delta_{JL}F_{kI} + \mu\,\delta_{IL}F_{kJ} + \left(\kappa - \frac{2}{3}\,\mu\right)\delta_{IJ}F_{kL}
    /// ```
    fn calculate_second_piola_kirchoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchoffTangentStiffness, ConstitutiveError> {
        let jacobian = deformation_gradient.determinant();
        if jacobian > 0.0 {
            let scaled_deformation_gradient_transpose =
                deformation_gradient.transpose() * self.get_shear_modulus();
            Ok(SecondPiolaKirchoffTangentStiffness::dyad_ik_jl(
                &scaled_deformation_gradient_transpose,
                &IDENTITY_00,
            ) + SecondPiolaKirchoffTangentStiffness::dyad_il_jk(
                &IDENTITY_00,
                &scaled_deformation_gradient_transpose,
            ) + SecondPiolaKirchoffTangentStiffness::dyad_ij_kl(
                &(IDENTITY_00 * (self.get_bulk_modulus() - TWO_THIRDS * self.get_shear_modulus())),
                deformation_gradient,
            ))
        } else {
            Err(ConstitutiveError::InvalidJacobian(
                jacobian,
                deformation_gradient.copy(),
                format!("{:?}", &self),
            ))
        }
    }
}

impl<'a> Hyperelastic<'a> for SaintVenantKirchoff<'a> {
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a(\mathbf{F}) = \mu\,\mathrm{tr}(\mathbf{E}^2) + \frac{1}{2}\left(\kappa - \frac{2}{3}\,\mu\right)\mathrm{tr}(\mathbf{E})^2
    /// ```
    fn calculate_helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        let jacobian = deformation_gradient.determinant();
        if jacobian > 0.0 {
            let strain = (self.calculate_right_cauchy_green_deformation(deformation_gradient)
                - IDENTITY_00)
                * 0.5;
            Ok(self.get_shear_modulus() * strain.squared_trace()
                + 0.5
                    * (self.get_bulk_modulus() - TWO_THIRDS * self.get_shear_modulus())
                    * strain.trace().powi(2))
        } else {
            Err(ConstitutiveError::InvalidJacobian(
                jacobian,
                deformation_gradient.copy(),
                format!("{:?}", &self),
            ))
        }
    }
}
