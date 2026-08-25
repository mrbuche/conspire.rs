#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{IDENTITY, Quantity, Rank2, TensorRank4},
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
    units::{EnergyDensity, Stress},
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct BlatzKo {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The mixing parameter $`f`$.
    pub mixing_parameter: Scalar,
}

impl BlatzKo {
    /// Returns the mixing parameter.
    pub fn mixing_parameter(&self) -> Scalar {
        self.mixing_parameter
    }
    fn n(&self) -> Scalar {
        0.5 * self.bulk_modulus().value() / self.shear_modulus().value() - 1.0 / 3.0
    }
}

impl Solid for BlatzKo {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Elastic for BlatzKo {
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let n = self.n();
        let f = self.mixing_parameter();
        let left_cauchy_green_deformation = deformation_gradient.left_cauchy_green();
        let inverse_left_cauchy_green_deformation = left_cauchy_green_deformation.inverse();
        let k = (1.0 - f) * jacobian.powf(2.0 * n) - f * jacobian.powf(-2.0 * n);
        Ok(
            (left_cauchy_green_deformation * f - inverse_left_cauchy_green_deformation * (1.0 - f)
                + IDENTITY * k)
                * (self.shear_modulus() / jacobian),
        )
    }
    #[doc = include_str!("cauchy_tangent_stiffness.md")]
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let n = self.n();
        let f = self.mixing_parameter();
        let left_cauchy_green_deformation = deformation_gradient.left_cauchy_green();
        let inverse_left_cauchy_green_deformation = left_cauchy_green_deformation.inverse();
        let jacobian_pos = jacobian.powf(2.0 * n);
        let jacobian_neg = jacobian.powf(-2.0 * n);
        let k = (1.0 - f) * jacobian_pos - f * jacobian_neg;
        let k_prime = 2.0 * n * ((1.0 - f) * jacobian_pos + f * jacobian_neg);
        let shear_modulus_over_jacobian = self.shear_modulus() / jacobian;
        Ok(((TensorRank4::dyad_ik_jl(&IDENTITY, deformation_gradient)
            + TensorRank4::dyad_il_jk(deformation_gradient, &IDENTITY)
            - TensorRank4::dyad_ij_kl(
                &left_cauchy_green_deformation,
                &inverse_transpose_deformation_gradient,
            ))
            * f
            + (TensorRank4::dyad_ik_jl(
                &inverse_left_cauchy_green_deformation,
                &inverse_transpose_deformation_gradient,
            ) + TensorRank4::dyad_il_jk(
                &inverse_transpose_deformation_gradient,
                &inverse_left_cauchy_green_deformation,
            ) + TensorRank4::dyad_ij_kl(
                &inverse_left_cauchy_green_deformation,
                &inverse_transpose_deformation_gradient,
            )) * (1.0 - f)
            + TensorRank4::dyad_ij_kl(&IDENTITY, &inverse_transpose_deformation_gradient)
                * (k_prime - k))
            * shear_modulus_over_jacobian)
    }
}

impl Hyperelastic for BlatzKo {
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let n = self.n();
        let f = self.mixing_parameter();
        let left_cauchy_green_deformation = deformation_gradient.left_cauchy_green();
        let first_invariant = left_cauchy_green_deformation.trace();
        let second_invariant = left_cauchy_green_deformation.second_invariant();
        let third_invariant = jacobian.powi(2);
        Ok(0.5
            * self.shear_modulus()
            * (f * (first_invariant - 3.0 + (third_invariant.powf(-n) - 1.0) / n)
                + (1.0 - f)
                    * (second_invariant / third_invariant - 3.0
                        + (third_invariant.powf(n) - 1.0) / n)))
    }
}
