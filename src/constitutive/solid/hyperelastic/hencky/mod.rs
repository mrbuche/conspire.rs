#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, TWO_THIRDS, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{ContractThirdFourthIndicesWithFirstSecondIndicesOf, IDENTITY, Rank2},
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct Hencky {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Scalar,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Scalar,
}

impl Solid for Hencky {
    fn bulk_modulus(&self) -> Scalar {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Scalar {
        self.shear_modulus
    }
}

impl Elastic for Hencky {
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let (deviatoric_strain, strain_trace) =
            (deformation_gradient.left_cauchy_green().logm()? * 0.5).deviatoric_and_trace();
        Ok(deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
            + IDENTITY * (self.bulk_modulus() * strain_trace / jacobian))
    }
    #[doc = include_str!("cauchy_tangent_stiffness.md")]
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let left_cauchy_green = deformation_gradient.left_cauchy_green();
        let (deviatoric_strain, strain_trace) =
            (left_cauchy_green.logm()? * 0.5).deviatoric_and_trace();
        let scaled_deformation_gradient = deformation_gradient * self.shear_modulus() / jacobian;
        Ok((left_cauchy_green
            .dlogm()?
            .contract_third_fourth_indices_with_first_second_indices_of(
                &(CauchyTangentStiffness::dyad_il_jk(&scaled_deformation_gradient, &IDENTITY)
                    + CauchyTangentStiffness::dyad_ik_jl(&IDENTITY, &scaled_deformation_gradient)),
            ))
            + (CauchyTangentStiffness::dyad_ij_kl(
                &(IDENTITY
                    * ((self.bulk_modulus() - TWO_THIRDS * self.shear_modulus()) / jacobian)
                    - deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
                    - IDENTITY * (self.bulk_modulus() * strain_trace / jacobian)),
                &deformation_gradient.inverse_transpose(),
            )))
    }
}

impl Hyperelastic for Hencky {
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let strain = deformation_gradient.left_cauchy_green().logm()? * 0.5;
        Ok(self.shear_modulus() * strain.squared_trace()
            + 0.5
                * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())
                * strain.trace().powi(2))
    }
}
