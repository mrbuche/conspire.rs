#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        Constitutive, ConstitutiveError,
        solid::{
            Solid, TWO_THIRDS,
            elastic_viscoplastic::{ElasticViscoplastic, Plastic, Viscoplastic},
            hyperelastic_viscoplastic::HyperelasticViscoplastic,
        },
    },
    math::{ContractThirdFourthIndicesWithFirstSecondIndicesOf, IDENTITY, Rank2},
    mechanics::{
        CauchyStress, CauchyTangentStiffness, CauchyTangentStiffnessElastic, Deformation,
        DeformationGradient, DeformationGradientPlastic, Scalar,
    },
};

#[doc = include_str!("doc.md")]
#[derive(Debug)]
pub struct Hencky {
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

impl Solid for Hencky {
    fn bulk_modulus(&self) -> &Scalar {
        &self.bulk_modulus
    }
    fn shear_modulus(&self) -> &Scalar {
        &self.shear_modulus
    }
}

impl Plastic for Hencky {
    fn initial_yield_stress(&self) -> &Scalar {
        &self.initial_yield_stress
    }
    fn hardening_slope(&self) -> &Scalar {
        &self.hardening_slope
    }
}

impl Viscoplastic for Hencky {
    fn rate_sensitivity(&self) -> &Scalar {
        &self.rate_sensitivity
    }
    fn reference_flow_rate(&self) -> &Scalar {
        &self.reference_flow_rate
    }
}

impl ElasticViscoplastic for Hencky {
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        let (deviatoric_strain_e, strain_trace_e) =
            (deformation_gradient_e.left_cauchy_green().logm() * 0.5).deviatoric_and_trace();
        Ok(
            deviatoric_strain_e * (2.0 * self.shear_modulus() / jacobian)
                + IDENTITY * (self.bulk_modulus() * strain_trace_e / jacobian),
        )
    }
    #[doc = include_str!("cauchy_tangent_stiffness.md")]
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let deformation_gradient_inverse_p = deformation_gradient_p.inverse();
        let deformation_gradient_e = deformation_gradient * &deformation_gradient_inverse_p;
        let left_cauchy_green_e = deformation_gradient_e.left_cauchy_green();
        let (deviatoric_strain_e, strain_trace_e) =
            (left_cauchy_green_e.logm() * 0.5).deviatoric_and_trace();
        let scaled_deformation_gradient_e =
            &deformation_gradient_e * self.shear_modulus() / jacobian;
        Ok((left_cauchy_green_e
            .dlogm()
            .contract_third_fourth_indices_with_first_second_indices_of(
                &(CauchyTangentStiffnessElastic::dyad_il_jk(
                    &scaled_deformation_gradient_e,
                    &IDENTITY,
                ) + CauchyTangentStiffnessElastic::dyad_ik_jl(
                    &IDENTITY,
                    &scaled_deformation_gradient_e,
                )),
            ))
            * deformation_gradient_inverse_p.transpose()
            + (CauchyTangentStiffness::dyad_ij_kl(
                &(IDENTITY
                    * ((self.bulk_modulus() - TWO_THIRDS * self.shear_modulus()) / jacobian)
                    - deviatoric_strain_e * (2.0 * self.shear_modulus() / jacobian)
                    - IDENTITY * (self.bulk_modulus() * strain_trace_e / jacobian)),
                &deformation_gradient.inverse_transpose(),
            )))
    }
}

impl HyperelasticViscoplastic for Hencky {
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<Scalar, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        let strain_e = deformation_gradient_e.left_cauchy_green().logm() * 0.5;
        Ok(self.shear_modulus() * strain_e.squared_trace()
            + 0.5
                * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())
                * strain_e.trace().powi(2))
    }
}
