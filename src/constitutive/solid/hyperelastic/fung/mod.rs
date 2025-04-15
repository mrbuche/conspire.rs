#[cfg(test)]
mod test;

use super::*;

#[doc = include_str!("model.md")]
#[derive(Debug)]
pub struct Fung<P> {
    parameters: P,
}

impl<P> Fung<P>
where
    P: Parameters,
{
    /// Returns the extra modulus.
    pub fn extra_modulus(&self) -> &Scalar {
        self.parameters.get(2)
    }
    /// Returns the exponent.
    pub fn exponent(&self) -> &Scalar {
        self.parameters.get(3)
    }
}

impl<P> Constitutive<P> for Fung<P>
where
    P: Parameters,
{
    fn new(parameters: P) -> Self {
        Self { parameters }
    }
}

impl<P> Solid<P> for Fung<P>
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

impl<P> Elastic<P> for Fung<P>
where
    P: Parameters,
{
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let isochoric_left_cauchy_green_deformation =
            deformation_gradient.left_cauchy_green() / jacobian.powf(TWO_THIRDS);
        let (
            deviatoric_isochoric_left_cauchy_green_deformation,
            isochoric_left_cauchy_green_deformation_trace,
        ) = isochoric_left_cauchy_green_deformation.deviatoric_and_trace();
        Ok(deviatoric_isochoric_left_cauchy_green_deformation
            * ((self.shear_modulus()
                + self.extra_modulus()
                    * ((self.exponent() * (isochoric_left_cauchy_green_deformation_trace - 3.0))
                        .exp()
                        - 1.0))
                / jacobian)
            + IDENTITY * self.bulk_modulus() * 0.5 * (jacobian - 1.0 / jacobian))
    }
    #[doc = include_str!("cauchy_tangent_stiffness.md")]
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let isochoric_left_cauchy_green_deformation =
            deformation_gradient.left_cauchy_green() / jacobian.powf(TWO_THIRDS);
        let (
            deviatoric_isochoric_left_cauchy_green_deformation,
            isochoric_left_cauchy_green_deformation_trace,
        ) = isochoric_left_cauchy_green_deformation.deviatoric_and_trace();
        let exponential =
            (self.exponent() * (isochoric_left_cauchy_green_deformation_trace - 3.0)).exp();
        let scaled_shear_modulus_0 = (self.shear_modulus()
            + self.extra_modulus() * (exponential - 1.0))
            / jacobian.powf(FIVE_THIRDS);
        Ok(
            (CauchyTangentStiffness::dyad_ik_jl(&IDENTITY, deformation_gradient)
                + CauchyTangentStiffness::dyad_il_jk(deformation_gradient, &IDENTITY)
                - CauchyTangentStiffness::dyad_ij_kl(&IDENTITY, deformation_gradient)
                    * (TWO_THIRDS))
                * scaled_shear_modulus_0
                + CauchyTangentStiffness::dyad_ij_kl(
                    &deviatoric_isochoric_left_cauchy_green_deformation,
                    &((&deviatoric_isochoric_left_cauchy_green_deformation
                        * &inverse_transpose_deformation_gradient)
                        * (2.0 * self.exponent() * self.extra_modulus() * exponential / jacobian)),
                )
                + CauchyTangentStiffness::dyad_ij_kl(
                    &(IDENTITY * (0.5 * self.bulk_modulus() * (jacobian + 1.0 / jacobian))
                        - deformation_gradient.left_cauchy_green().deviatoric()
                            * (scaled_shear_modulus_0 * FIVE_THIRDS)),
                    &inverse_transpose_deformation_gradient,
                ),
        )
    }
}

impl<P> Hyperelastic<P> for Fung<P>
where
    P: Parameters,
{
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let scalar_term =
            deformation_gradient.left_cauchy_green().trace() / jacobian.powf(TWO_THIRDS) - 3.0;
        Ok(0.5
            * ((self.shear_modulus() - self.extra_modulus()) * scalar_term
                + self.extra_modulus() / self.exponent()
                    * ((self.exponent() * scalar_term).exp() - 1.0)
                + self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln())))
    }
}
