#[cfg(test)]
mod test;
use crate::math::Quantity;
use crate::math::TensorRank4;

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
    math::{IDENTITY_22, Rank2, TensorArray},
    mechanics::{
        Deformation, DeformationGradient, DeformationGradientPlastic, MandelStressElastic, Scalar,
        SecondPiolaKirchhoffStress, SecondPiolaKirchhoffTangentStiffness,
    },
    units::{EnergyDensity, Rate, Stress},
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct SaintVenantKirchhoff {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Scalar,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Scalar,
    /// The initial yield stress $`Y_0`$.
    pub yield_stress: Scalar,
    /// The isotropic hardening slope $`H`$.
    pub hardening_slope: Scalar,
    /// The rate sensitivity parameter $`m`$.
    pub rate_sensitivity: Scalar,
    /// The reference flow rate $`d_0`$.
    pub reference_flow_rate: Scalar,
}

impl Solid for SaintVenantKirchhoff {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus.into()
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus.into()
    }
}

impl Plastic for SaintVenantKirchhoff {
    fn initial_yield_stress(&self) -> Quantity<Stress> {
        self.yield_stress.into()
    }
    fn hardening_slope(&self) -> Quantity<Stress> {
        self.hardening_slope.into()
    }
}

impl Viscoplastic<Quantity> for SaintVenantKirchhoff {
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
        self.reference_flow_rate.into()
    }
}

impl ElasticPlasticOrViscoplastic for SaintVenantKirchhoff {
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
        Ok((TensorRank4::dyad_ik_jl(&quantity_2, &scaled_quantity_1)
            + TensorRank4::dyad_il_jk(&scaled_quantity_1, &quantity_2))
            + TensorRank4::dyad_ij_kl(
                &(quantity_1 * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())),
                &quantity_2.transpose(),
            ))
    }
}

impl ElasticViscoplastic<Quantity> for SaintVenantKirchhoff {}

impl HyperelasticViscoplastic<Quantity> for SaintVenantKirchhoff {
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        let strain = (deformation_gradient_e.right_cauchy_green() - IDENTITY_22) * 0.5;
        Ok(self.shear_modulus() * strain.squared_trace()
            + 0.5
                * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())
                * strain.trace().powi(2))
    }
}
