#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::{plastic::Plastic, viscoplastic::Viscoplastic},
        solid::{
            Solid, TWO_THIRDS, elastic_viscoplastic::ElasticViscoplastic,
            hyperelastic_viscoplastic::HyperelasticViscoplastic,
        },
    },
    math::{IDENTITY_22, Rank2},
    mechanics::{
        Deformation, DeformationGradient, DeformationGradientPlastic, Scalar,
        SecondPiolaKirchhoffStress, SecondPiolaKirchhoffTangentStiffness,
    },
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct SaintVenantKirchhoff {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Scalar,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Scalar,
    /// The initial yield stress $`Y_0`$.
    pub initial_yield_stress: Scalar,
    /// The isotropic hardening slope $`H`$.
    pub hardening_slope: Scalar,
    /// The rate sensitivity parameter $`m`$.
    pub rate_sensitivity: Scalar,
    /// The reference flow rate $`d_0`$.
    pub reference_flow_rate: Scalar,
}

impl Solid for SaintVenantKirchhoff {
    fn bulk_modulus(&self) -> Scalar {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Scalar {
        self.shear_modulus
    }
}

impl Plastic for SaintVenantKirchhoff {
    fn initial_yield_stress(&self) -> Scalar {
        self.initial_yield_stress
    }
    fn hardening_slope(&self) -> Scalar {
        self.hardening_slope
    }
}

impl Viscoplastic for SaintVenantKirchhoff {
    fn rate_sensitivity(&self) -> Scalar {
        self.rate_sensitivity
    }
    fn reference_flow_rate(&self) -> Scalar {
        self.reference_flow_rate
    }
}

impl ElasticViscoplastic for SaintVenantKirchhoff {
    #[doc = include_str!("second_piola_kirchhoff_stress.md")]
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let deformation_gradient_inverse_p = deformation_gradient_p.inverse();
        let deformation_gradient_e = deformation_gradient * &deformation_gradient_inverse_p;
        let left_cauchy_green_inverse_p = deformation_gradient_inverse_p.left_cauchy_green();
        let (deviatoric_strain, strain_trace) =
            ((deformation_gradient_e.right_cauchy_green() - IDENTITY_22) * 0.5)
                .deviatoric_and_trace();
        Ok(&deformation_gradient_inverse_p
            * deviatoric_strain
            * deformation_gradient_inverse_p.transpose()
            * (2.0 * self.shear_modulus())
            + left_cauchy_green_inverse_p * (self.bulk_modulus() * strain_trace))
    }
    #[doc = include_str!("second_piola_kirchhoff_tangent_stiffness.md")]
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let deformation_gradient_inverse_p = deformation_gradient_p.inverse();
        let deformation_gradient_e = deformation_gradient * &deformation_gradient_inverse_p;
        let quantity_1 = deformation_gradient_inverse_p.left_cauchy_green();
        let quantity_2 = deformation_gradient_inverse_p * deformation_gradient_e.transpose();
        let scaled_quantity_1 = &quantity_1 * self.shear_modulus();
        Ok(
            (SecondPiolaKirchhoffTangentStiffness::dyad_ik_jl(&quantity_2, &scaled_quantity_1)
                + SecondPiolaKirchhoffTangentStiffness::dyad_il_jk(
                    &scaled_quantity_1,
                    &quantity_2,
                ))
                + SecondPiolaKirchhoffTangentStiffness::dyad_ij_kl(
                    &(quantity_1 * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())),
                    &quantity_2.transpose(),
                ),
        )
    }
}

impl HyperelasticViscoplastic for SaintVenantKirchhoff {
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<Scalar, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        let strain = (deformation_gradient_e.right_cauchy_green() - IDENTITY_22) * 0.5;
        Ok(self.shear_modulus() * strain.squared_trace()
            + 0.5
                * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())
                * strain.trace().powi(2))
    }
}
