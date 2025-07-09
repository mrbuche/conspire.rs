#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        Constitutive, ConstitutiveError, Parameters,
        solid::{Solid, TWO_THIRDS, elastic::Elastic},
    },
    math::{IDENTITY, Rank2},
    mechanics::{CauchyStress, CauchyTangentStiffness, DeformationGradient, Scalar},
};

#[doc = include_str!("doc.md")]
#[derive(Debug)]
pub struct AlmansiHamel<P> {
    parameters: P,
}

impl<P> Constitutive<P> for AlmansiHamel<P>
where
    P: Parameters,
{
    fn new(parameters: P) -> Self {
        Self { parameters }
    }
}

impl<P> Solid for AlmansiHamel<P>
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

impl<P> Elastic for AlmansiHamel<P>
where
    P: Parameters,
{
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_deformation_gradient = deformation_gradient.inverse();
        let strain = (IDENTITY
            - inverse_deformation_gradient.transpose() * &inverse_deformation_gradient)
            * 0.5;
        let (deviatoric_strain, strain_trace) = strain.deviatoric_and_trace();
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
        let inverse_left_cauchy_green_deformation = &inverse_transpose_deformation_gradient
            * inverse_transpose_deformation_gradient.transpose();
        let strain = (IDENTITY - &inverse_left_cauchy_green_deformation) * 0.5;
        let (deviatoric_strain, strain_trace) = strain.deviatoric_and_trace();
        Ok((CauchyTangentStiffness::dyad_il_jk(
            &inverse_transpose_deformation_gradient,
            &inverse_left_cauchy_green_deformation,
        ) + CauchyTangentStiffness::dyad_ik_jl(
            &inverse_left_cauchy_green_deformation,
            &inverse_transpose_deformation_gradient,
        )) * (self.shear_modulus() / jacobian)
            + CauchyTangentStiffness::dyad_ij_kl(
                &IDENTITY,
                &(inverse_left_cauchy_green_deformation
                    * &inverse_transpose_deformation_gradient
                    * ((self.bulk_modulus() - self.shear_modulus() * TWO_THIRDS) / jacobian)),
            )
            - CauchyTangentStiffness::dyad_ij_kl(
                &(deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
                    + IDENTITY * (self.bulk_modulus() * strain_trace / jacobian)),
                &inverse_transpose_deformation_gradient,
            ))
    }
}
