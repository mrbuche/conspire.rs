#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::{
            hyperviscous::{Hyperviscous, TWO_THIRDS},
            viscous::Viscous,
        },
    },
    math::{ContractWith, IDENTITY_00, Quantity, Rank2, TensorRank4},
    mechanics::{
        DeformationGradient, DeformationGradientRate, SecondPiolaKirchhoffRateTangentStiffness,
        SecondPiolaKirchhoffStress,
    },
    units::{Dissipation, Viscosity},
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct SaintVenantKirchhoff {
    /// The bulk viscosity $`\zeta`$.
    pub bulk_viscosity: Quantity<Viscosity>,
    /// The shear viscosity $`\eta`$.
    pub shear_viscosity: Quantity<Viscosity>,
}

impl Viscous for SaintVenantKirchhoff {
    fn bulk_viscosity(&self) -> Quantity<Viscosity> {
        self.bulk_viscosity
    }
    fn shear_viscosity(&self) -> Quantity<Viscosity> {
        self.shear_viscosity
    }
    #[doc = include_str!("viscous_second_piola_kirchhoff_stress.md")]
    fn viscous_second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        let strain_rate_term = deformation_gradient_rate.transpose() * deformation_gradient;
        let (deviatoric_strain_rate, strain_rate_trace) =
            ((&strain_rate_term + strain_rate_term.transpose()) * 0.5).deviatoric_and_trace();
        Ok(deviatoric_strain_rate * (2.0 * self.shear_viscosity())
            + IDENTITY_00 * (self.bulk_viscosity() * strain_rate_trace))
    }
    #[doc = include_str!("viscous_second_piola_kirchhoff_rate_tangent_stiffness.md")]
    fn viscous_second_piola_kirchhoff_rate_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        _: &DeformationGradientRate,
    ) -> Result<SecondPiolaKirchhoffRateTangentStiffness, ConstitutiveError> {
        let scaled_deformation_gradient_transpose =
            deformation_gradient.transpose() * self.shear_viscosity();
        Ok(
            TensorRank4::dyad_ik_jl(&scaled_deformation_gradient_transpose, &IDENTITY_00)
                + TensorRank4::dyad_il_jk(&IDENTITY_00, &scaled_deformation_gradient_transpose)
                + TensorRank4::dyad_ij_kl(
                    &(IDENTITY_00 * (self.bulk_viscosity() - TWO_THIRDS * self.shear_viscosity())),
                    deformation_gradient,
                ),
        )
    }
}

impl Hyperviscous for SaintVenantKirchhoff {
    #[doc = include_str!("viscous_dissipation.md")]
    fn viscous_dissipation(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        let strain_rate_term = deformation_gradient_rate.transpose() * deformation_gradient;
        let strain_rate = (&strain_rate_term + strain_rate_term.transpose()) * 0.5;
        let strain_rate_trace = strain_rate.trace();
        Ok(
            (&strain_rate * self.shear_viscosity()).contract_with(&strain_rate)
                + (self.bulk_viscosity() - TWO_THIRDS * self.shear_viscosity())
                    * strain_rate_trace
                    * strain_rate_trace
                    * 0.5,
        )
    }
}
