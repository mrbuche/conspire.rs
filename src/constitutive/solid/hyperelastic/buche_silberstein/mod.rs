#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{FIVE_THIRDS, Solid, TWO_THIRDS, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{
        IDENTITY, Quantity, Rank2, TensorRank4,
        special::{langevin, langevin_derivative, sinhc},
    },
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
    units::{EnergyDensity, Stress},
};
use std::f64::consts::TAU;

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct BucheSilberstein {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The number of links $`N_b`$.
    pub number_of_links: Scalar,
    /// The nondimensional link stiffness $`\varkappa`$.
    pub link_stiffness: Scalar,
}

impl BucheSilberstein {
    /// Returns the number of links.
    pub fn number_of_links(&self) -> Scalar {
        self.number_of_links
    }
    /// Returns the nondimensional link stiffness.
    pub fn link_stiffness(&self) -> Scalar {
        self.link_stiffness
    }
    /// Returns the nondimensional single-chain force $`\eta`$ at a nondimensional
    /// end-to-end length $`\gamma`$, i.e. the inverse of the reduced extensible
    /// freely-jointed chain relation
    ///
    /// ```math
    /// \gamma(\eta) = \mathcal{L}(\eta) + \frac{\eta}{\varkappa}.
    /// ```
    ///
    /// Cohen's rational inverse Langevin, composed with the linear bond term,
    /// collapses to a single cubic in the orientational stretch `u = γ − η/ϰ`,
    ///
    /// ```math
    /// (\varkappa + 1)u^3 - \varkappa\gamma u^2 - (\varkappa + 3)u + \varkappa\gamma = 0,
    /// ```
    ///
    /// with exactly one root in $`(0,1)`$.  That root (Cardano; $`p<0`$ here, so the
    /// trigonometric branch) seeds a few Newton iterations on the exact relation.
    fn nondimensional_force(&self, gamma: Scalar) -> Scalar {
        let kappa = self.link_stiffness();
        let (a, b, c, d) = (kappa + 1.0, -kappa * gamma, -(kappa + 3.0), kappa * gamma);
        let shift = b / (3.0 * a);
        let p = (3.0 * a * c - b * b) / (3.0 * a * a);
        let q = (2.0 * b.powi(3) - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a.powi(3));
        let m = 2.0 * (-p / 3.0).sqrt();
        let theta = (3.0 * q / (p * m)).clamp(-1.0, 1.0).acos() / 3.0;
        let mut eta = kappa
            * (gamma
                - (0..3)
                    .map(|k| m * (theta - TAU * (k as Scalar) / 3.0).cos() - shift)
                    .find(|root| (0.0..1.0).contains(root))
                    .unwrap_or_else(|| (kappa * gamma / (kappa + 3.0)).min(1.0 - 1e-12)));
        for _ in 0..3 {
            eta -= (langevin(eta) + eta / kappa - gamma) / (langevin_derivative(eta) + 1.0 / kappa);
        }
        eta.max(0.0)
    }
}

impl Solid for BucheSilberstein {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Elastic for BucheSilberstein {
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let (deviatoric_isochoric_left_cauchy_green_deformation, isochoric_trace) =
            (deformation_gradient.left_cauchy_green() / jacobian.powf(TWO_THIRDS))
                .deviatoric_and_trace();
        let gamma = (isochoric_trace / 3.0 / self.number_of_links())
            .sqrt()
            .value();
        let gamma_0 = (1.0 / self.number_of_links()).sqrt();
        Ok(deviatoric_isochoric_left_cauchy_green_deformation
            * (self.shear_modulus() * self.nondimensional_force(gamma)
                / self.nondimensional_force(gamma_0)
                * gamma_0
                / gamma
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
        let left_cauchy_green_deformation = deformation_gradient.left_cauchy_green();
        let deviatoric_left_cauchy_green_deformation = left_cauchy_green_deformation.deviatoric();
        let (deviatoric_isochoric_left_cauchy_green_deformation, isochoric_trace) =
            (left_cauchy_green_deformation / jacobian.powf(TWO_THIRDS)).deviatoric_and_trace();
        let gamma = (isochoric_trace / 3.0 / self.number_of_links())
            .sqrt()
            .value();
        let gamma_0 = (1.0 / self.number_of_links()).sqrt();
        let kappa = self.link_stiffness();
        let eta = self.nondimensional_force(gamma);
        let scaled_shear_modulus =
            gamma_0 / self.nondimensional_force(gamma_0) * self.shear_modulus() * eta
                / gamma
                / jacobian.powf(FIVE_THIRDS);
        let scaled_deviatoric_isochoric_left_cauchy_green_deformation =
            deviatoric_left_cauchy_green_deformation * scaled_shear_modulus;
        // d(eta)/d(gamma) = 1 / (L'(eta) + 1/kappa)  for  gamma = L(eta) + eta/kappa
        let term = TensorRank4::dyad_ij_kl(
            &scaled_deviatoric_isochoric_left_cauchy_green_deformation,
            &(deviatoric_isochoric_left_cauchy_green_deformation
                * &inverse_transpose_deformation_gradient
                * ((1.0 / eta / (langevin_derivative(eta) + 1.0 / kappa) - 1.0 / gamma)
                    / 3.0
                    / self.number_of_links()
                    / gamma)),
        );
        Ok((TensorRank4::dyad_ik_jl(&IDENTITY, deformation_gradient)
            + TensorRank4::dyad_il_jk(deformation_gradient, &IDENTITY)
            - TensorRank4::dyad_ij_kl(&IDENTITY, deformation_gradient) * (TWO_THIRDS))
            * scaled_shear_modulus
            + TensorRank4::dyad_ij_kl(
                &(IDENTITY * (0.5 * self.bulk_modulus() * (jacobian + 1.0 / jacobian))
                    - scaled_deviatoric_isochoric_left_cauchy_green_deformation * (FIVE_THIRDS)),
                &inverse_transpose_deformation_gradient,
            )
            + term)
    }
}

impl Hyperelastic for BucheSilberstein {
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let isochoric_left_cauchy_green_deformation =
            deformation_gradient.left_cauchy_green() / jacobian.powf(TWO_THIRDS);
        let gamma =
            (isochoric_left_cauchy_green_deformation.trace() / 3.0 / self.number_of_links())
                .sqrt()
                .value();
        let gamma_0 = (1.0 / self.number_of_links()).sqrt();
        let kappa = self.link_stiffness();
        let eta = self.nondimensional_force(gamma);
        let eta_0 = self.nondimensional_force(gamma_0);
        // psi*(gamma) = gamma eta - ln[sinh(eta)/eta] - eta^2 / (2 kappa)
        let psi = |g: Scalar, e: Scalar| g * e - sinhc(e).ln() - e.powi(2) / (2.0 * kappa);
        Ok(3.0 * gamma_0 / eta_0
            * self.shear_modulus()
            * self.number_of_links()
            * (psi(gamma, eta) - psi(gamma_0, eta_0))
            + 0.5 * self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln()))
    }
}
