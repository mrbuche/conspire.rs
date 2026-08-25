#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{FIVE_THIRDS, Solid, TWO_THIRDS, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{IDENTITY, Quantity, Rank2, TensorRank4},
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
    units::{EnergyDensity, Modulus, Stress},
};
use std::iter::once;

const SEVEN_THIRDS: Scalar = 7.0 / 3.0;

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct Yeoh {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The extra moduli $`\mu_n`$ for $`n=2\ldots N`$.
    pub extra_moduli: Vec<Quantity<Modulus>>,
}

impl Yeoh {
    /// Returns the extra moduli.
    pub fn extra_moduli(&self) -> impl Iterator<Item = Quantity<Modulus>> {
        self.extra_moduli.iter().copied()
    }
}

impl Solid for Yeoh {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Elastic for Yeoh {
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
            * once(self.shear_modulus())
                .chain(self.extra_moduli())
                .enumerate()
                .map(|(n, modulus)| modulus * (((n as Scalar) + 1.0) * scalar_term.powi(n as i32)))
                .sum::<Quantity<Modulus>>()
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
        let scaled_modulus = once(self.shear_modulus())
            .chain(self.extra_moduli())
            .enumerate()
            .map(|(n, modulus)| modulus * (((n as Scalar) + 1.0) * scalar_term.powi(n as i32)))
            .sum::<Quantity<Modulus>>()
            / jacobian.powf(FIVE_THIRDS);
        let deviatoric_left_cauchy_green_deformation = left_cauchy_green_deformation.deviatoric();
        let last_term = TensorRank4::dyad_ij_kl(
            &deviatoric_left_cauchy_green_deformation,
            &((left_cauchy_green_deformation.deviatoric()
                * &inverse_transpose_deformation_gradient)
                * (self
                    .extra_moduli()
                    .enumerate()
                    .map(|(n, modulus)| {
                        modulus
                            * (2.0
                                * ((n as Scalar) + 2.0)
                                * ((n as Scalar) + 1.0)
                                * scalar_term.powi(n as i32))
                    })
                    .sum::<Quantity<Modulus>>()
                    / jacobian.powf(SEVEN_THIRDS))),
        );
        Ok((TensorRank4::dyad_ik_jl(&IDENTITY, deformation_gradient)
            + TensorRank4::dyad_il_jk(deformation_gradient, &IDENTITY)
            - TensorRank4::dyad_ij_kl(&IDENTITY, deformation_gradient) * (TWO_THIRDS))
            * scaled_modulus
            + TensorRank4::dyad_ij_kl(
                &(IDENTITY * (self.bulk_modulus() * 0.5 * (jacobian + 1.0 / jacobian))
                    - deviatoric_left_cauchy_green_deformation * (scaled_modulus * FIVE_THIRDS)),
                &inverse_transpose_deformation_gradient,
            )
            + last_term)
    }
}

impl Hyperelastic for Yeoh {
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let scalar_term =
            deformation_gradient.left_cauchy_green().trace() / jacobian.powf(TWO_THIRDS) - 3.0;
        Ok(0.5
            * (once(self.shear_modulus())
                .chain(self.extra_moduli())
                .enumerate()
                .map(|(n, modulus)| modulus * scalar_term.powi((n + 1) as i32))
                .sum::<Quantity<Modulus>>()
                + self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln())))
    }
}
