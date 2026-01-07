#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{FIVE_THIRDS, Solid, TWO_THIRDS, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{IDENTITY, Rank2},
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct Fung {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Scalar,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Scalar,
    /// The extra modulus $`\mu_m`$.
    pub extra_modulus: Scalar,
    /// The exponent $`c`$.
    pub exponent: Scalar,
}

impl Fung {
    /// Returns the extra modulus.
    pub fn extra_modulus(&self) -> Scalar {
        self.extra_modulus
    }
    /// Returns the exponent.
    pub fn exponent(&self) -> Scalar {
        self.exponent
    }
}

impl Solid for Fung {
    fn bulk_modulus(&self) -> Scalar {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Scalar {
        self.shear_modulus
    }
}

impl Elastic for Fung {
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

impl Hyperelastic for Fung {
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
