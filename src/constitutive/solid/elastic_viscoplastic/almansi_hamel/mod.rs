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
        },
    },
    math::{IDENTITY, Quantity, Rank2, TensorArray, TensorRank4},
    mechanics::{
        CauchyStress, CauchyTangentStiffness, DeformationGradient, DeformationGradientPlastic,
        MandelStressElastic, Scalar,
    },
    units::{Rate, Stress},
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct AlmansiHamel {
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

impl Solid for AlmansiHamel {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Plastic for AlmansiHamel {
    fn initial_yield_stress(&self) -> Quantity<Stress> {
        self.yield_stress
    }
    fn hardening_slope(&self) -> Quantity<Stress> {
        self.hardening_slope
    }
}

impl Viscoplastic<Quantity> for AlmansiHamel {
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

impl ElasticPlasticOrViscoplastic for AlmansiHamel {
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        let inverse_deformation_gradient_e = deformation_gradient_e.inverse();
        let (deviatoric_strain_e, strain_trace_e) = ((IDENTITY
            - inverse_deformation_gradient_e.transpose() * &inverse_deformation_gradient_e)
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
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        let inverse_transpose_deformation_gradient_e = deformation_gradient_e.inverse_transpose();
        let inverse_left_cauchy_green_deformation_e = &inverse_transpose_deformation_gradient_e
            * inverse_transpose_deformation_gradient_e.transpose();
        let scaled_inverse_left_cauchy_green_deformation_e =
            &inverse_left_cauchy_green_deformation_e * (self.shear_modulus() / jacobian);
        let strain_e = (IDENTITY - &inverse_left_cauchy_green_deformation_e) * 0.5;
        let (deviatoric_strain_e, strain_trace_e) = strain_e.deviatoric_and_trace();
        Ok((TensorRank4::dyad_il_jk(
            &inverse_transpose_deformation_gradient,
            &scaled_inverse_left_cauchy_green_deformation_e,
        ) + TensorRank4::dyad_ik_jl(
            &scaled_inverse_left_cauchy_green_deformation_e,
            &inverse_transpose_deformation_gradient,
        )) + TensorRank4::dyad_ij_kl(
            &IDENTITY,
            &(inverse_left_cauchy_green_deformation_e
                * &inverse_transpose_deformation_gradient
                * ((self.bulk_modulus() - self.shear_modulus() * TWO_THIRDS) / jacobian)),
        ) - TensorRank4::dyad_ij_kl(
            &(deviatoric_strain_e * (2.0 * self.shear_modulus() / jacobian)
                + IDENTITY * (self.bulk_modulus() * strain_trace_e / jacobian)),
            &inverse_transpose_deformation_gradient,
        ))
    }
}

impl ElasticViscoplastic<Quantity> for AlmansiHamel {}
