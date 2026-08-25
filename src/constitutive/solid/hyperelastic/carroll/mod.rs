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
pub struct Carroll {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The linear modulus $`\mu_1`$.
    pub linear_modulus: Quantity<Stress>,
    /// The quartic modulus $`\mu_2`$.
    pub quartic_modulus: Quantity<Stress>,
    /// The second-invariant modulus $`\mu_3`$.
    pub second_invariant_modulus: Quantity<Stress>,
}

impl Carroll {
    /// Returns the quartic modulus.
    pub fn quartic_modulus(&self) -> Quantity<Stress> {
        self.quartic_modulus
    }
    /// Returns the second-invariant modulus.
    pub fn second_invariant_modulus(&self) -> Quantity<Stress> {
        self.second_invariant_modulus
    }
}

impl Solid for Carroll {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.linear_modulus * 2.0
            + self.quartic_modulus * 216.0
            + self.second_invariant_modulus / 3.0_f64.sqrt()
    }
}

impl Elastic for Carroll {
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let isochoric_left_cauchy_green_deformation =
            deformation_gradient.left_cauchy_green() / jacobian.powf(TWO_THIRDS);
        let first_invariant = isochoric_left_cauchy_green_deformation.trace();
        let second_invariant = isochoric_left_cauchy_green_deformation.second_invariant();
        let dw_di1 = self.linear_modulus + self.quartic_modulus * (4.0 * first_invariant.powi(3));
        let dw_di2 = self.second_invariant_modulus / (2.0 * second_invariant.sqrt());
        Ok(
            ((isochoric_left_cauchy_green_deformation.deviatoric() * (dw_di1 * 2.0)
                - isochoric_left_cauchy_green_deformation
                    .inverse()
                    .deviatoric()
                    * (dw_di2 * 2.0))
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
        let isochoric_left_cauchy_green_deformation =
            deformation_gradient.left_cauchy_green() / jacobian.powf(TWO_THIRDS);
        let first_invariant = isochoric_left_cauchy_green_deformation.trace();
        let second_invariant = isochoric_left_cauchy_green_deformation.second_invariant();
        let dw_di1 = self.linear_modulus + self.quartic_modulus * (4.0 * first_invariant.powi(3));
        let dw_di2 = self.second_invariant_modulus / (2.0 * second_invariant.sqrt());
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
            &((deviatoric_inverse_isochoric_left_cauchy_green_deformation.clone() * TWO_THIRDS)
                * &inverse_transpose_deformation_gradient),
        );
        let d_first_invariant_dw_di1_df = deformation_gradient
            * (48.0 * self.quartic_modulus * first_invariant.powi(2) / jacobian.powf(TWO_THIRDS))
            - inverse_transpose_deformation_gradient.clone()
                * (16.0 * self.quartic_modulus * first_invariant.powi(3));
        let identity_i1_minus_b =
            IDENTITY * first_invariant - isochoric_left_cauchy_green_deformation.clone();
        let d_second_invariant_dw_di2_df = (identity_i1_minus_b * deformation_gradient)
            * (-2.0 * dw_di2 / second_invariant / jacobian.powf(TWO_THIRDS))
            + inverse_transpose_deformation_gradient.clone() * (dw_di2 * FOUR_THIRDS);
        let extra_term_1 = TensorRank4::dyad_ij_kl(
            &isochoric_left_cauchy_green_deformation.deviatoric(),
            &d_first_invariant_dw_di1_df,
        ) / jacobian;
        let extra_term_2 = TensorRank4::dyad_ij_kl(
            &deviatoric_inverse_isochoric_left_cauchy_green_deformation,
            &d_second_invariant_dw_di2_df,
        ) / jacobian;
        Ok((TensorRank4::dyad_ik_jl(&IDENTITY, deformation_gradient)
            + TensorRank4::dyad_il_jk(deformation_gradient, &IDENTITY)
            - TensorRank4::dyad_ij_kl(&IDENTITY, deformation_gradient) * (TWO_THIRDS))
            * scaled_dw_di1
            + TensorRank4::dyad_ij_kl(
                &(IDENTITY * (0.5 * self.bulk_modulus() * (jacobian + 1.0 / jacobian))
                    - deformation_gradient.left_cauchy_green().deviatoric()
                        * (scaled_dw_di1 * FIVE_THIRDS)),
                &inverse_transpose_deformation_gradient,
            )
            - (term_1 + term_2 - term_3) * (dw_di2 * 2.0) / jacobian
            + extra_term_1
            - extra_term_2)
    }
}

impl Hyperelastic for Carroll {
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
        Ok(self.linear_modulus * (first_invariant - 3.0)
            + self.quartic_modulus * (first_invariant.powi(4) - 81.0)
            + self.second_invariant_modulus * (second_invariant.sqrt() - 3.0_f64.sqrt())
            + self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln()) * 0.5)
    }
}
