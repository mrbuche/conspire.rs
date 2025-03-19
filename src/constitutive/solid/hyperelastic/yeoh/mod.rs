#[cfg(test)]
mod test;

use super::*;

const SEVEN_THIRDS: Scalar = 7.0 / 3.0;

#[doc = include_str!("model.md")]
#[derive(Debug)]
pub struct Yeoh<'a> {
    parameters: Parameters<'a>,
}

impl Yeoh<'_> {
    /// Returns an array of the moduli.
    pub fn moduli(&self) -> &[Scalar] {
        &self.parameters[1..]
    }
    /// Returns an array of the extra moduli.
    pub fn extra_moduli(&self) -> &[Scalar] {
        &self.parameters[2..]
    }
}

impl<'a> Constitutive<'a> for Yeoh<'a> {
    fn new(parameters: Parameters<'a>) -> Self {
        Self { parameters }
    }
}

impl<'a> Solid<'a> for Yeoh<'a> {
    fn bulk_modulus(&self) -> &Scalar {
        &self.parameters[0]
    }
    fn shear_modulus(&self) -> &Scalar {
        &self.parameters[1]
    }
}

impl<'a> Elastic<'a> for Yeoh<'a> {
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let (deviatoric_left_cauchy_green_deformation, left_cauchy_green_deformation_trace) =
            deformation_gradient
                .left_cauchy_green()
                .deviatoric_and_trace();
        let scalar_term = left_cauchy_green_deformation_trace / jacobian.powf(TWO_THIRDS) - 3.0;
        Ok(deviatoric_left_cauchy_green_deformation
            * self
                .moduli()
                .iter()
                .enumerate()
                .map(|(n, modulus)| ((n as Scalar) + 1.0) * modulus * scalar_term.powi(n as i32))
                .sum::<Scalar>()
            / jacobian.powf(FIVE_THIRDS)
            + IDENTITY * self.bulk_modulus() * 0.5 * (jacobian - 1.0 / jacobian))
    }
    #[doc = include_str!("cauchy_tangent_stiffness.md")]
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let left_cauchy_green_deformation = deformation_gradient.left_cauchy_green();
        let scalar_term = left_cauchy_green_deformation.trace() / jacobian.powf(TWO_THIRDS) - 3.0;
        let scaled_modulus = self
            .moduli()
            .iter()
            .enumerate()
            .map(|(n, modulus)| ((n as Scalar) + 1.0) * modulus * scalar_term.powi(n as i32))
            .sum::<Scalar>()
            / jacobian.powf(FIVE_THIRDS);
        let deviatoric_left_cauchy_green_deformation = left_cauchy_green_deformation.deviatoric();
        let last_term = CauchyTangentStiffness::dyad_ij_kl(
            &deviatoric_left_cauchy_green_deformation,
            &((left_cauchy_green_deformation.deviatoric()
                * &inverse_transpose_deformation_gradient)
                * (2.0
                    * self
                        .extra_moduli()
                        .iter()
                        .enumerate()
                        .map(|(n, modulus)| {
                            ((n as Scalar) + 2.0)
                                * ((n as Scalar) + 1.0)
                                * modulus
                                * scalar_term.powi(n as i32)
                        })
                        .sum::<Scalar>()
                    / jacobian.powf(SEVEN_THIRDS))),
        );
        Ok(
            (CauchyTangentStiffness::dyad_ik_jl(&IDENTITY, deformation_gradient)
                + CauchyTangentStiffness::dyad_il_jk(deformation_gradient, &IDENTITY)
                - CauchyTangentStiffness::dyad_ij_kl(&IDENTITY, deformation_gradient)
                    * (TWO_THIRDS))
                * scaled_modulus
                + CauchyTangentStiffness::dyad_ij_kl(
                    &(IDENTITY * (0.5 * self.bulk_modulus() * (jacobian + 1.0 / jacobian))
                        - deviatoric_left_cauchy_green_deformation
                            * (scaled_modulus * FIVE_THIRDS)),
                    &inverse_transpose_deformation_gradient,
                )
                + last_term,
        )
    }
}

impl<'a> Hyperelastic<'a> for Yeoh<'a> {
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let scalar_term =
            deformation_gradient.left_cauchy_green().trace() / jacobian.powf(TWO_THIRDS) - 3.0;
        Ok(0.5
            * (self
                .moduli()
                .iter()
                .enumerate()
                .map(|(n, modulus)| modulus * scalar_term.powi((n + 1) as i32))
                .sum::<Scalar>()
                + self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln())))
    }
}
