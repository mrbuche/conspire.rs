#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, TWO_THIRDS, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{IDENTITY_00, Quantity, Rank2, TensorRank4},
    mechanics::{
        Deformation, DeformationGradient, SecondPiolaKirchhoffStress,
        SecondPiolaKirchhoffTangentStiffness,
    },
    units::{EnergyDensity, Stress},
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct SaintVenantKirchhoff {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
}

impl Solid for SaintVenantKirchhoff {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Elastic for SaintVenantKirchhoff {
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
        Ok(
            TensorRank4::dyad_ik_jl(&scaled_deformation_gradient_transpose, &IDENTITY_00)
                + TensorRank4::dyad_il_jk(&IDENTITY_00, &scaled_deformation_gradient_transpose)
                + TensorRank4::dyad_ij_kl(
                    &(IDENTITY_00 * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())),
                    deformation_gradient,
                ),
        )
    }
}

impl Hyperelastic for SaintVenantKirchhoff {
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let strain = (deformation_gradient.right_cauchy_green() - IDENTITY_00) * 0.5;
        Ok(self.shear_modulus() * strain.squared_trace()
            + 0.5
                * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())
                * strain.trace().powi(2))
    }
}
