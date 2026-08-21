#[cfg(test)]
mod test;

use crate::math::{ContractWith, Quantity};
use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::viscous::Viscous,
        solid::{
            Solid, TWO_THIRDS, elastic_hyperviscous::ElasticHyperviscous,
            viscoelastic::Viscoelastic,
        },
    },
    math::{IDENTITY, Rank2, TensorRank4},
    mechanics::{
        CauchyRateTangentStiffness, CauchyStress, DeformationGradient, DeformationGradientRate,
    },
    units::{Dissipation, Stress, Viscosity},
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct AlmansiHamel {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The bulk viscosity $`\zeta`$.
    pub bulk_viscosity: Quantity<Viscosity>,
    /// The shear viscosity $`\eta`$.
    pub shear_viscosity: Quantity<Viscosity>,
}

impl Solid for AlmansiHamel {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Viscous for AlmansiHamel {
    fn bulk_viscosity(&self) -> Quantity<Viscosity> {
        self.bulk_viscosity
    }
    fn shear_viscosity(&self) -> Quantity<Viscosity> {
        self.shear_viscosity
    }
}

impl Viscoelastic for AlmansiHamel {
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let bulk_modulus = self.bulk_modulus();
        let shear_modulus = self.shear_modulus();
        let bulk_viscosity = self.bulk_viscosity();
        let shear_viscosity = self.shear_viscosity();
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_deformation_gradient = deformation_gradient.inverse();
        let strain = (IDENTITY
            - inverse_deformation_gradient.transpose() * &inverse_deformation_gradient)
            * 0.5;
        let (deviatoric_strain, strain_trace) = strain.deviatoric_and_trace();
        let velocity_gradient = deformation_gradient_rate * inverse_deformation_gradient;
        let strain_rate = (&velocity_gradient + velocity_gradient.transpose()) * 0.5;
        let (deviatoric_strain_rate, strain_rate_trace) = strain_rate.deviatoric_and_trace();
        Ok(deviatoric_strain * (2.0 * shear_modulus / jacobian)
            + deviatoric_strain_rate * (2.0 * shear_viscosity / jacobian)
            + IDENTITY
                * ((bulk_modulus * strain_trace + bulk_viscosity * strain_rate_trace) / jacobian))
    }
    #[doc = include_str!("cauchy_rate_tangent_stiffness.md")]
    fn cauchy_rate_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        _: &DeformationGradientRate,
    ) -> Result<CauchyRateTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let scaled_deformation_gradient_inverse_transpose =
            &deformation_gradient_inverse_transpose * self.shear_viscosity() / jacobian;
        Ok(
            TensorRank4::dyad_ik_jl(&IDENTITY, &scaled_deformation_gradient_inverse_transpose)
                + TensorRank4::dyad_il_jk(
                    &scaled_deformation_gradient_inverse_transpose,
                    &IDENTITY,
                )
                + TensorRank4::dyad_ij_kl(
                    &(IDENTITY
                        * ((self.bulk_viscosity() - TWO_THIRDS * self.shear_viscosity())
                            / jacobian)),
                    &deformation_gradient_inverse_transpose,
                ),
        )
    }
}

impl ElasticHyperviscous for AlmansiHamel {
    #[doc = include_str!("viscous_dissipation.md")]
    fn viscous_dissipation(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
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
