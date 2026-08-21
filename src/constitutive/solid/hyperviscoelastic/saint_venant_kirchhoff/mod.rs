#[cfg(test)]
mod test;
use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::viscous::Viscous,
        solid::{
            Solid, TWO_THIRDS, elastic_hyperviscous::ElasticHyperviscous,
            hyperviscoelastic::Hyperviscoelastic, viscoelastic::Viscoelastic,
        },
    },
    math::{ContractWith, IDENTITY_00, Quantity, Rank2, TensorRank4},
    mechanics::{
        Deformation, DeformationGradient, DeformationGradientRate,
        SecondPiolaKirchhoffRateTangentStiffness, SecondPiolaKirchhoffStress,
    },
    units::{Dissipation, EnergyDensity, Stress, Viscosity},
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct SaintVenantKirchhoff {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The bulk viscosity $`\zeta`$.
    pub bulk_viscosity: Quantity<Viscosity>,
    /// The shear viscosity $`\eta`$.
    pub shear_viscosity: Quantity<Viscosity>,
}

impl Solid for SaintVenantKirchhoff {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Viscous for SaintVenantKirchhoff {
    fn bulk_viscosity(&self) -> Quantity<Viscosity> {
        self.bulk_viscosity
    }
    fn shear_viscosity(&self) -> Quantity<Viscosity> {
        self.shear_viscosity
    }
}

impl Viscoelastic for SaintVenantKirchhoff {
    #[doc = include_str!("second_piola_kirchhoff_stress.md")]
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let (deviatoric_strain, strain_trace) =
            ((deformation_gradient.right_cauchy_green() - IDENTITY_00) * 0.5)
                .deviatoric_and_trace();
        let first_term = deformation_gradient_rate.transpose() * deformation_gradient;
        let (deviatoric_strain_rate, strain_rate_trace) =
            ((&first_term + first_term.transpose()) * 0.5).deviatoric_and_trace();
        let bulk_modulus = self.bulk_modulus();
        let shear_modulus = self.shear_modulus();
        let bulk_viscosity = self.bulk_viscosity();
        let shear_viscosity = self.shear_viscosity();
        Ok(deviatoric_strain * (2.0 * shear_modulus)
            + deviatoric_strain_rate * (2.0 * shear_viscosity)
            + IDENTITY_00 * (bulk_modulus * strain_trace + bulk_viscosity * strain_rate_trace))
    }
    #[doc = include_str!("second_piola_kirchhoff_rate_tangent_stiffness.md")]
    fn second_piola_kirchhoff_rate_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        _: &DeformationGradientRate,
    ) -> Result<SecondPiolaKirchhoffRateTangentStiffness, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
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

impl ElasticHyperviscous for SaintVenantKirchhoff {
    #[doc = include_str!("viscous_dissipation.md")]
    fn viscous_dissipation(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let first_term = deformation_gradient_rate.transpose() * deformation_gradient;
        let strain_rate = (&first_term + first_term.transpose()) * 0.5;
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

impl Hyperviscoelastic for SaintVenantKirchhoff {
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
