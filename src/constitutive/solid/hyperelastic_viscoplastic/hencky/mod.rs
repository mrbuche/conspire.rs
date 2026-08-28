#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::{
            plastic::Plastic,
            viscoplastic::{
                Viscoplastic, ViscoplasticEvolution, ViscoplasticStateVariables,
                default_plastic_evolution,
            },
        },
        solid::{
            Solid, TWO_THIRDS,
            elastic_viscoplastic::{ElasticPlasticOrViscoplastic, ElasticViscoplastic},
            hyperelastic_viscoplastic::HyperelasticViscoplastic,
        },
    },
    math::{
        ContractThirdFourthWithFirstSecond, IDENTITY, Quantity, Rank2, TensorArray, TensorRank4,
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient,
        DeformationGradientPlastic, MandelStressElastic, Scalar,
    },
    units::{EnergyDensity, Rate, Stress},
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct Hencky {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The initial yield stress $`Y_0`$.
    pub yield_stress: Quantity<Stress>,
    /// The isotropic hardening slope $`H`$.
    pub hardening_slope: Quantity<Stress>,
    /// The rate sensitivity parameter $`m`$.
    pub rate_sensitivity: Scalar,
    /// The reference flow rate $`d_0`$.
    pub reference_flow_rate: Quantity<Rate>,
}

impl Solid for Hencky {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Plastic for Hencky {
    fn initial_yield_stress(&self) -> Quantity<Stress> {
        self.yield_stress
    }
    fn hardening_slope(&self) -> Quantity<Stress> {
        self.hardening_slope
    }
}

impl Viscoplastic<Quantity> for Hencky {
    fn initial_state(&self) -> ViscoplasticStateVariables<Quantity> {
        (DeformationGradientPlastic::identity(), Quantity::default()).into()
    }
    fn plastic_evolution(
        &self,
        mandel_stress: MandelStressElastic,
        state_variables: &ViscoplasticStateVariables<Quantity>,
    ) -> Result<ViscoplasticEvolution<Quantity>, ConstitutiveError> {
        default_plastic_evolution(self, mandel_stress, state_variables)
    }
    fn rate_sensitivity(&self) -> Scalar {
        self.rate_sensitivity
    }
    fn reference_flow_rate(&self) -> Quantity<Rate> {
        self.reference_flow_rate
    }
}

impl ElasticPlasticOrViscoplastic for Hencky {
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        let (deviatoric_strain_e, strain_trace_e) = (deformation_gradient_e
            .left_cauchy_green()
            .logm()
            .map_err(|error| ConstitutiveError::upstream(error, self))?
            * 0.5)
            .deviatoric_and_trace();
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
        let (deviatoric_strain_e, strain_trace_e) = (left_cauchy_green_e
            .logm()
            .map_err(|error| ConstitutiveError::upstream(error, self))?
            * 0.5)
            .deviatoric_and_trace();
        let scaled_deformation_gradient_e =
            &deformation_gradient_e * self.shear_modulus() / jacobian;
        Ok((left_cauchy_green_e
            .dlogm()
            .map_err(|error| ConstitutiveError::upstream(error, self))?
            .contract_third_fourth_with_first_second(
                &(TensorRank4::dyad_il_jk(&scaled_deformation_gradient_e, &IDENTITY)
                    + TensorRank4::dyad_ik_jl(&IDENTITY, &scaled_deformation_gradient_e)),
            ))
            * deformation_gradient_inverse_p.transpose()
            + (TensorRank4::dyad_ij_kl(
                &(IDENTITY
                    * ((self.bulk_modulus() - TWO_THIRDS * self.shear_modulus()) / jacobian)
                    - deviatoric_strain_e * (2.0 * self.shear_modulus() / jacobian)
                    - IDENTITY * (self.bulk_modulus() * strain_trace_e / jacobian)),
                &deformation_gradient.inverse_transpose(),
            )))
    }
}

impl ElasticViscoplastic<Quantity> for Hencky {}

impl HyperelasticViscoplastic<Quantity> for Hencky {
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        let strain_e = deformation_gradient_e
            .left_cauchy_green()
            .logm()
            .map_err(|error| ConstitutiveError::upstream(error, self))?
            * 0.5;
        Ok(self.shear_modulus() * strain_e.squared_trace()
            + 0.5
                * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())
                * strain_e.trace().powi(2))
    }
}
