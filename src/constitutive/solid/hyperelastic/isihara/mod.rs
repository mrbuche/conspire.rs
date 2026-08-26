#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{FIVE_THIRDS, Solid, TWO_THIRDS, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{IDENTITY, Quantity, Rank2, TensorRank4},
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
    units::{EnergyDensity, Stress},
};

const FOUR_THIRDS: Scalar = 4.0 / 3.0;

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct Isihara {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The extra modulus $`\mu_m`$.
    pub extra_modulus: Quantity<Stress>,
    /// The quadratic modulus $`\mu_q`$.
    pub quadratic_modulus: Quantity<Stress>,
}

impl Isihara {
    /// Returns the extra modulus.
    pub fn extra_modulus(&self) -> Quantity<Stress> {
        self.extra_modulus
    }
    /// Returns the quadratic modulus.
    pub fn quadratic_modulus(&self) -> Quantity<Stress> {
        self.quadratic_modulus
    }
}

impl Solid for Isihara {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Elastic for Isihara {
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let isochoric_left_cauchy_green_deformation =
            deformation_gradient.left_cauchy_green() / jacobian.powf(TWO_THIRDS);
        let first_invariant = isochoric_left_cauchy_green_deformation.trace();
        let dw_di1 = 0.5 * (self.shear_modulus() - self.extra_modulus())
            + self.quadratic_modulus() * (first_invariant - 3.0);
        Ok(
            ((isochoric_left_cauchy_green_deformation.deviatoric() * (dw_di1 * 2.0)
                - isochoric_left_cauchy_green_deformation
                    .inverse()
                    .deviatoric()
                    * self.extra_modulus())
                + IDENTITY * (self.bulk_modulus() * 0.5 * (jacobian.powi(2) - 1.0)))
                / jacobian,
        )
    }
    #[doc = include_str!("cauchy_tangent_stiffness.md")]
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let left_cauchy_green_deformation = deformation_gradient.left_cauchy_green();
        let isochoric_left_cauchy_green_deformation =
            left_cauchy_green_deformation.clone() / jacobian.powf(TWO_THIRDS);
        let first_invariant = isochoric_left_cauchy_green_deformation.trace();
        let dw_di1 = 0.5 * (self.shear_modulus() - self.extra_modulus())
            + self.quadratic_modulus() * (first_invariant - 3.0);
        let scaled_dw_di1 = dw_di1 * 2.0 / jacobian.powf(FIVE_THIRDS);
        let inverse_isochoric_left_cauchy_green_deformation =
            isochoric_left_cauchy_green_deformation.inverse();
        let deviatoric_inverse_isochoric_left_cauchy_green_deformation =
            inverse_isochoric_left_cauchy_green_deformation.deviatoric();
        let term_1 = TensorRank4::dyad_ij_kl(
            &inverse_isochoric_left_cauchy_green_deformation,
            &inverse_transpose_deformation_gradient,
        ) * TWO_THIRDS
            - TensorRank4::dyad_ik_jl(
                &inverse_isochoric_left_cauchy_green_deformation,
                &inverse_transpose_deformation_gradient,
            )
            - TensorRank4::dyad_il_jk(
                &inverse_transpose_deformation_gradient,
                &inverse_isochoric_left_cauchy_green_deformation,
            );
        let term_3 = TensorRank4::dyad_ij_kl(
            &deviatoric_inverse_isochoric_left_cauchy_green_deformation,
            &inverse_transpose_deformation_gradient,
        );
        let term_2 = TensorRank4::dyad_ij_kl(
            &IDENTITY,
            &((deviatoric_inverse_isochoric_left_cauchy_green_deformation * TWO_THIRDS)
                * &inverse_transpose_deformation_gradient),
        );
        let d_first_invariant_dw_di1_df = deformation_gradient
            * (4.0 * self.quadratic_modulus() / jacobian.powf(TWO_THIRDS))
            - inverse_transpose_deformation_gradient.clone()
                * (FOUR_THIRDS * self.quadratic_modulus() * first_invariant);
        let extra_term_1 = TensorRank4::dyad_ij_kl(
            &isochoric_left_cauchy_green_deformation.deviatoric(),
            &d_first_invariant_dw_di1_df,
        ) / jacobian;
        Ok((TensorRank4::dyad_ik_jl(&IDENTITY, deformation_gradient)
            + TensorRank4::dyad_il_jk(deformation_gradient, &IDENTITY)
            - TensorRank4::dyad_ij_kl(&IDENTITY, deformation_gradient) * (TWO_THIRDS))
            * scaled_dw_di1
            + TensorRank4::dyad_ij_kl(
                &(IDENTITY * (0.5 * self.bulk_modulus() * (jacobian + 1.0 / jacobian))
                    - left_cauchy_green_deformation.deviatoric() * (scaled_dw_di1 * FIVE_THIRDS)),
                &inverse_transpose_deformation_gradient,
            )
            - (term_1 + term_2 - term_3) * self.extra_modulus() / jacobian
            + extra_term_1)
    }
}

impl Hyperelastic for Isihara {
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let isochoric_left_cauchy_green_deformation =
            deformation_gradient.left_cauchy_green() / jacobian.powf(TWO_THIRDS);
        let first_invariant = isochoric_left_cauchy_green_deformation.trace();
        let second_invariant = isochoric_left_cauchy_green_deformation.second_invariant();
        Ok(0.5
            * ((self.shear_modulus() - self.extra_modulus()) * (first_invariant - 3.0)
                + self.extra_modulus() * (second_invariant - 3.0)
                + self.quadratic_modulus() * (first_invariant - 3.0).powi(2)
                + self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln())))
    }
}
