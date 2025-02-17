#[cfg(test)]
mod test;

use super::*;

#[doc = include_str!("model.md")]
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
    #[doc = include_str!("second_piola_kirchoff_stress.md")]
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
    #[doc = include_str!("second_piola_kirchoff_tangent_stiffness.md")]
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
    #[doc = include_str!("helmholtz_free_energy_density.md")]
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
