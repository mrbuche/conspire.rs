#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        Constitutive, ConstitutiveError, Parameters,
        solid::{FIVE_THIRDS, Solid, TWO_THIRDS, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{IDENTITY, Rank2},
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
};

#[doc = include_str!("doc.md")]
#[derive(Debug)]
pub struct NeoHookean<P> {
    parameters: P,
}

impl<P> Constitutive<P> for NeoHookean<P>
where
    P: Parameters,
{
    fn new(parameters: P) -> Self {
        Self { parameters }
    }
}

impl<P> Solid for NeoHookean<P>
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

impl<P> Elastic for NeoHookean<P>
where
    P: Parameters,
{
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        Ok(
            deformation_gradient.left_cauchy_green().deviatoric() / jacobian.powf(FIVE_THIRDS)
                * self.shear_modulus()
                + IDENTITY * self.bulk_modulus() * 0.5 * (jacobian - 1.0 / jacobian),
        )
    }
    #[doc = include_str!("cauchy_tangent_stiffness.md")]
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let scaled_shear_modulus = self.shear_modulus() / jacobian.powf(FIVE_THIRDS);
        Ok(
            (CauchyTangentStiffness::dyad_ik_jl(&IDENTITY, deformation_gradient)
                + CauchyTangentStiffness::dyad_il_jk(deformation_gradient, &IDENTITY)
                - CauchyTangentStiffness::dyad_ij_kl(&IDENTITY, deformation_gradient)
                    * (TWO_THIRDS))
                * scaled_shear_modulus
                + CauchyTangentStiffness::dyad_ij_kl(
                    &(IDENTITY * (0.5 * self.bulk_modulus() * (jacobian + 1.0 / jacobian))
                        - deformation_gradient.left_cauchy_green().deviatoric()
                            * (scaled_shear_modulus * FIVE_THIRDS)),
                    &inverse_transpose_deformation_gradient,
                ),
        )
    }
}

impl<P> Hyperelastic for NeoHookean<P>
where
    P: Parameters,
{
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        Ok(0.5
            * (self.shear_modulus()
                * (deformation_gradient.left_cauchy_green().trace() / jacobian.powf(TWO_THIRDS)
                    - 3.0)
                + self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln())))
    }
}
