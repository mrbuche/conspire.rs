#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        Constitutive, ConstitutiveError, Parameters,
        solid::{Solid, TWO_THIRDS, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{IDENTITY, Rank2},
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
};

#[doc = include_str!("doc.md")]
#[derive(Debug)]
pub struct Hencky<P> {
    parameters: P,
}

impl<P> Constitutive<P> for Hencky<P>
where
    P: Parameters,
{
    fn new(parameters: P) -> Self {
        Self { parameters }
    }
}

impl<P> Solid for Hencky<P>
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

impl<P> Elastic for Hencky<P>
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
            (deformation_gradient.left_cauchy_green().logm() * 0.5).deviatoric_and_trace();
        Ok(deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
            + IDENTITY * (self.bulk_modulus() * strain_trace / jacobian))
    }
    #[doc = include_str!("cauchy_tangent_stiffness.md")]
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        // let jacobian = self.jacobian(deformation_gradient)?;
        // let (deviatoric_strain, strain_trace) =
        //     (deformation_gradient.left_cauchy_green().logm() * 0.5).deviatoric_and_trace();
        // let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        // let mu_over_j = self.shear_modulus() / jacobian;
        // let kappa_over_j = self.bulk_modulus() / jacobian;
        // Ok(CauchyTangentStiffness::dyad_ik_jl(
        //     &(IDENTITY * mu_over_j),
        //     &inverse_transpose_deformation_gradient,
        // ) + CauchyTangentStiffness::dyad_il_jk(
        //     &inverse_transpose_deformation_gradient,
        //     &(IDENTITY * mu_over_j),
        // ) + CauchyTangentStiffness::dyad_ij_kl(
        //     &(IDENTITY * ((kappa_over_j - mu_over_j * TWO_THIRDS) - kappa_over_j * strain_trace)
        //         - deviatoric_strain * (2.0 * mu_over_j)),
        //     &inverse_transpose_deformation_gradient,
        // ))
        let jacobian = self.jacobian(deformation_gradient)?;
        let left_cauchy_green = deformation_gradient.left_cauchy_green();
        let inverse_left_cauchy_green = left_cauchy_green.inverse();
        let (deviatoric_strain, strain_trace) =
            (left_cauchy_green.logm() * 0.5).deviatoric_and_trace();
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        Ok(
            (CauchyTangentStiffness::dyad_ik_jl(&inverse_left_cauchy_green, deformation_gradient)
                + CauchyTangentStiffness::dyad_il_jk(
                    deformation_gradient,
                    &inverse_left_cauchy_green,
                )
                + CauchyTangentStiffness::dyad_il_jk(
                    &inverse_transpose_deformation_gradient,
                    &IDENTITY,
                )
                + CauchyTangentStiffness::dyad_ik_jl(
                    &IDENTITY,
                    &inverse_transpose_deformation_gradient,
                ))
                * self.shear_modulus()
                / jacobian
                / 2.0
                + (CauchyTangentStiffness::dyad_ij_kl(
                    &(IDENTITY
                        * ((self.bulk_modulus() - TWO_THIRDS * self.shear_modulus()) / jacobian)),
                    &inverse_transpose_deformation_gradient,
                ))
                - (CauchyTangentStiffness::dyad_ij_kl(
                    &(deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
                        + IDENTITY * (self.bulk_modulus() * strain_trace / jacobian)),
                    &inverse_transpose_deformation_gradient,
                )),
        )
    }
}

impl<P> Hyperelastic for Hencky<P>
where
    P: Parameters,
{
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let strain = deformation_gradient.left_cauchy_green().logm() * 0.5;
        Ok(self.shear_modulus() * strain.squared_trace()
            + 0.5
                * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())
                * strain.trace().powi(2))
    }
}
