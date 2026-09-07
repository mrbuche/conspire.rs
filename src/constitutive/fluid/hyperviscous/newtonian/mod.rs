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
    math::{ContractWith, IDENTITY, Quantity, Rank2, TensorRank4},
    mechanics::{
        CauchyRateTangentStiffness, CauchyStress, DeformationGradient, DeformationGradientRate,
    },
    units::{Dissipation, Viscosity},
};

/// The Newtonian viscous fluid constitutive model.
#[derive(Clone, Debug)]
pub struct Newtonian {
    /// The bulk viscosity $`\zeta`$.
    pub bulk_viscosity: Quantity<Viscosity>,
    /// The shear viscosity $`\eta`$.
    pub shear_viscosity: Quantity<Viscosity>,
}

impl Viscous for Newtonian {
    fn bulk_viscosity(&self) -> Quantity<Viscosity> {
        self.bulk_viscosity
    }
    fn shear_viscosity(&self) -> Quantity<Viscosity> {
        self.shear_viscosity
    }
    fn viscous_cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let velocity_gradient = deformation_gradient_rate * deformation_gradient.inverse();
        let (deviatoric_strain_rate, strain_rate_trace) =
            ((&velocity_gradient + velocity_gradient.transpose()) * 0.5).deviatoric_and_trace();
        Ok(deviatoric_strain_rate * (2.0 * self.shear_viscosity())
            + IDENTITY * (self.bulk_viscosity() * strain_rate_trace))
    }
    fn viscous_cauchy_rate_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        _: &DeformationGradientRate,
    ) -> Result<CauchyRateTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let scaled_deformation_gradient_inverse_transpose =
            &deformation_gradient_inverse_transpose * self.shear_viscosity();
        Ok(
            TensorRank4::dyad_ik_jl(&IDENTITY, &scaled_deformation_gradient_inverse_transpose)
                + TensorRank4::dyad_il_jk(
                    &scaled_deformation_gradient_inverse_transpose,
                    &IDENTITY,
                )
                + TensorRank4::dyad_ij_kl(
                    &(IDENTITY * (self.bulk_viscosity() - TWO_THIRDS * self.shear_viscosity())),
                    &deformation_gradient_inverse_transpose,
                ),
        )
    }
}

impl Hyperviscous for Newtonian {
    fn viscous_dissipation(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        let velocity_gradient = deformation_gradient_rate * deformation_gradient.inverse();
        let strain_rate = (&velocity_gradient + velocity_gradient.transpose()) * 0.5;
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
