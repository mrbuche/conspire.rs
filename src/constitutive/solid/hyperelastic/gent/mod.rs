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

#[doc = include_str!("model.md")]
#[derive(Debug)]
pub struct Gent<P> {
    parameters: P,
}

impl<P> Gent<P>
where
    P: Parameters,
{
    /// Returns the extensibility.
    pub fn extensibility(&self) -> &Scalar {
        self.parameters.get(2)
    }
}

impl<P> Constitutive<P> for Gent<P>
where
    P: Parameters,
{
    fn new(parameters: P) -> Self {
        Self { parameters }
    }
}

impl<P> Solid for Gent<P>
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

impl<P> Elastic for Gent<P>
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
        let denominator =
            self.extensibility() - isochoric_left_cauchy_green_deformation_trace + 3.0;
        if denominator <= 0.0 {
            Err(ConstitutiveError::Custom(
                "Maximum extensibility reached.".to_string(),
                deformation_gradient.clone(),
                format!("{:?}", &self),
            ))
        } else {
            Ok((deviatoric_isochoric_left_cauchy_green_deformation
                * self.shear_modulus()
                * self.extensibility()
                / jacobian)
                / denominator
                + IDENTITY * self.bulk_modulus() * 0.5 * (jacobian - 1.0 / jacobian))
        }
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
        let denominator =
            self.extensibility() - isochoric_left_cauchy_green_deformation_trace + 3.0;
        if denominator <= 0.0 {
            Err(ConstitutiveError::Custom(
                "Maximum extensibility reached.".to_string(),
                deformation_gradient.clone(),
                format!("{:?}", &self),
            ))
        } else {
            let prefactor = self.shear_modulus() * self.extensibility() / jacobian / denominator;
            Ok(
                (CauchyTangentStiffness::dyad_ik_jl(&IDENTITY, deformation_gradient)
                    + CauchyTangentStiffness::dyad_il_jk(deformation_gradient, &IDENTITY)
                    - CauchyTangentStiffness::dyad_ij_kl(&IDENTITY, deformation_gradient)
                        * (TWO_THIRDS)
                    + CauchyTangentStiffness::dyad_ij_kl(
                        &deviatoric_isochoric_left_cauchy_green_deformation,
                        deformation_gradient,
                    ) * (2.0 / denominator))
                    * (prefactor / jacobian.powf(TWO_THIRDS))
                    + CauchyTangentStiffness::dyad_ij_kl(
                        &(IDENTITY * (0.5 * self.bulk_modulus() * (jacobian + 1.0 / jacobian))
                            - deviatoric_isochoric_left_cauchy_green_deformation
                                * prefactor
                                * ((5.0
                                    + 2.0 * isochoric_left_cauchy_green_deformation_trace
                                        / denominator)
                                    / 3.0)),
                        &inverse_transpose_deformation_gradient,
                    ),
            )
        }
    }
}

impl<P> Hyperelastic for Gent<P>
where
    P: Parameters,
{
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let factor = (deformation_gradient.left_cauchy_green().trace() / jacobian.powf(TWO_THIRDS)
            - 3.0)
            / self.extensibility();
        if factor >= 1.0 {
            Err(ConstitutiveError::Custom(
                "Maximum extensibility reached.".to_string(),
                deformation_gradient.clone(),
                format!("{:?}", &self),
            ))
        } else {
            Ok(0.5
                * (-self.shear_modulus() * self.extensibility() * (1.0 - factor).ln()
                    + self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln())))
        }
    }
}
