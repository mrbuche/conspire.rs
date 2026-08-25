#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{ContractThirdFourthWithFirstSecond, IDENTITY, Quantity, Rank2, TensorRank4},
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
    units::{EnergyDensity, Stress},
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct Ogden {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The moduli $`\mu_n`$.
    pub moduli: Vec<Quantity<Stress>>,
    /// The exponents $`\alpha_n`$.
    pub exponents: Vec<Scalar>,
}

impl Solid for Ogden {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.moduli
            .iter()
            .zip(self.exponents.iter())
            .map(|(modulus, exponent)| *modulus * exponent)
            .sum::<Quantity<Stress>>()
            * 0.5
    }
}

impl Elastic for Ogden {
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let left_cauchy_green = deformation_gradient.left_cauchy_green();
        let mut cauchy_stress = IDENTITY * self.bulk_modulus() * 0.5 * (jacobian - 1.0 / jacobian);
        for (modulus, exponent) in self.moduli.iter().zip(self.exponents.iter()) {
            let scaling = *modulus / jacobian.powf(exponent / 3.0 + 1.0);
            cauchy_stress += left_cauchy_green.powm(exponent / 2.0)?.deviatoric() * scaling;
        }
        Ok(cauchy_stress)
    }
    #[doc = include_str!("cauchy_tangent_stiffness.md")]
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let left_cauchy_green = deformation_gradient.left_cauchy_green();
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let mut cauchy_tangent_stiffness = TensorRank4::dyad_ij_kl(
            &(IDENTITY * (self.bulk_modulus() * 0.5 * (jacobian + 1.0 / jacobian))),
            &inverse_transpose_deformation_gradient,
        );
        for (modulus, exponent) in self.moduli.iter().zip(self.exponents.iter()) {
            let half_exponent = exponent / 2.0;
            let scaling = *modulus / jacobian.powf(exponent / 3.0 + 1.0);
            let scaled_deformation_gradient = deformation_gradient * scaling;
            let raw = left_cauchy_green
                .dpowm(half_exponent)?
                .contract_third_fourth_with_first_second(
                    &(TensorRank4::dyad_il_jk(&scaled_deformation_gradient, &IDENTITY)
                        + TensorRank4::dyad_ik_jl(&IDENTITY, &scaled_deformation_gradient)),
                );
            let trace_term = left_cauchy_green.powm(half_exponent - 1.0)?
                * deformation_gradient
                * (scaling * half_exponent * 2.0);
            let cauchy_stress_n = left_cauchy_green.powm(half_exponent)?.deviatoric() * scaling;
            cauchy_tangent_stiffness += raw
                - TensorRank4::dyad_ij_kl(&(IDENTITY * (1.0 / 3.0)), &trace_term)
                - TensorRank4::dyad_ij_kl(
                    &(cauchy_stress_n * (exponent / 3.0 + 1.0)),
                    &inverse_transpose_deformation_gradient,
                );
        }
        Ok(cauchy_tangent_stiffness)
    }
}

impl Hyperelastic for Ogden {
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let left_cauchy_green = deformation_gradient.left_cauchy_green();
        let mut helmholtz_free_energy_density =
            self.bulk_modulus() * 0.5 * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln());
        for (modulus, exponent) in self.moduli.iter().zip(self.exponents.iter()) {
            helmholtz_free_energy_density += (*modulus / *exponent)
                * (left_cauchy_green.powm(exponent / 2.0)?.trace() / jacobian.powf(exponent / 3.0)
                    - 3.0);
        }
        Ok(helmholtz_free_energy_density)
    }
}
