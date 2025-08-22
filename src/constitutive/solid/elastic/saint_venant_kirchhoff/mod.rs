#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        Constitutive, ConstitutiveError, Parameters,
        solid::{Solid, TWO_THIRDS, elastic::Elastic},
    },
    math::{IDENTITY, Rank2},
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
};

#[doc = include_str!("doc.md")]
#[derive(Debug)]
pub struct SaintVenantKirchhoff<P> {
    parameters: P,
}

impl<P> Constitutive<P> for SaintVenantKirchhoff<P>
where
    P: Parameters,
{
    fn new(parameters: P) -> Self {
        Self { parameters }
    }
}

impl<P> Solid for SaintVenantKirchhoff<P>
where
    P: Parameters,
{
    fn bulk_modulus(&self) -> &Scalar {
        self.parameters.get(0)
    }
    fn shear_modulus(&self) -> &Scalar {
        self.parameters.get(1)
    }
}

impl<P> Elastic for SaintVenantKirchhoff<P>
where
    P: Parameters,
{
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let (deviatoric_strain, strain_trace) =
            ((deformation_gradient.left_cauchy_green() - IDENTITY) * 0.5).deviatoric_and_trace();
        Ok(deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
            + IDENTITY * (self.bulk_modulus() * strain_trace / jacobian))
    }
    #[doc = include_str!("cauchy_tangent_stiffness.md")]
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let scaled_deformation_gradient = deformation_gradient * (self.shear_modulus() / jacobian);
        let (deviatoric_strain, strain_trace) =
            ((deformation_gradient.left_cauchy_green() - IDENTITY) * 0.5).deviatoric_and_trace();
        Ok(
            (CauchyTangentStiffness::dyad_il_jk(&scaled_deformation_gradient, &IDENTITY)
                + CauchyTangentStiffness::dyad_ik_jl(&IDENTITY, &scaled_deformation_gradient))
                + CauchyTangentStiffness::dyad_ij_kl(
                    &IDENTITY,
                    &(deformation_gradient
                        * ((self.bulk_modulus() - self.shear_modulus() * TWO_THIRDS) / jacobian)),
                )
                - CauchyTangentStiffness::dyad_ij_kl(
                    &(deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
                        + IDENTITY * (self.bulk_modulus() * strain_trace / jacobian)),
                    &inverse_transpose_deformation_gradient,
                ),
        )
    }
}
