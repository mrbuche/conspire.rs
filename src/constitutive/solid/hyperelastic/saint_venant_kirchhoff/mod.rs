#[cfg(test)]
mod test;

use super::*;

#[doc = include_str!("model.md")]
#[derive(Debug)]
pub struct SaintVenantKirchhoff<P> {
    parameters: P,
}

impl<P> Constitutive<P> for SaintVenantKirchhoff<P> where P: Parameters {
    fn new(parameters: P) -> Self {
        Self { parameters }
    }
}

impl<P> Solid<P> for SaintVenantKirchhoff<P> where P: Parameters {
    fn bulk_modulus(&self) -> &Scalar {
        self.parameters.get(0)
    }
    fn shear_modulus(&self) -> &Scalar {
        self.parameters.get(1)
    }
}

impl<P> Elastic<P> for SaintVenantKirchhoff<P> where P: Parameters {
    #[doc = include_str!("second_piola_kirchhoff_stress.md")]
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let (deviatoric_strain, strain_trace) =
            ((deformation_gradient.right_cauchy_green() - IDENTITY_00) * 0.5)
                .deviatoric_and_trace();
        Ok(deviatoric_strain * (2.0 * self.shear_modulus())
            + IDENTITY_00 * (self.bulk_modulus() * strain_trace))
    }
    #[doc = include_str!("second_piola_kirchhoff_tangent_stiffness.md")]
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let scaled_deformation_gradient_transpose =
            deformation_gradient.transpose() * self.shear_modulus();
        Ok(SecondPiolaKirchhoffTangentStiffness::dyad_ik_jl(
            &scaled_deformation_gradient_transpose,
            &IDENTITY_00,
        ) + SecondPiolaKirchhoffTangentStiffness::dyad_il_jk(
            &IDENTITY_00,
            &scaled_deformation_gradient_transpose,
        ) + SecondPiolaKirchhoffTangentStiffness::dyad_ij_kl(
            &(IDENTITY_00 * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())),
            deformation_gradient,
        ))
    }
}

impl<P> Hyperelastic<P> for SaintVenantKirchhoff<P> where P: Parameters {
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let strain = (deformation_gradient.right_cauchy_green() - IDENTITY_00) * 0.5;
        Ok(self.shear_modulus() * strain.squared_trace()
            + 0.5
                * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())
                * strain.trace().powi(2))
    }
}
