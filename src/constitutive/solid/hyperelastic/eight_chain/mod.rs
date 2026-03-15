#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{FIVE_THIRDS, Solid, TWO_THIRDS, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{IDENTITY, Rank2},
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
    physics::molecular::single_chain::Thermodynamics as SingleChainThermodynamics,
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct EightChain<T>
where
    T: SingleChainThermodynamics,
{
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Scalar,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Scalar,
    /// The single-chain model.
    pub single_chain_model: T,
}

// should there be an initial length field specification? (otherwise need number of links N_b)
// could require a second trait that has number_of_links
// but what if want to use something like WLC?
// so then if sqrt(N) is average (RMS?) length, the trait should give that
// but initial length also would be normalized by l, which also isnt always there
// so maybe just stick with discrete models for now

impl<T> EightChain<T>
where
    T: SingleChainThermodynamics,
{
    /// ???
    fn nondimensional_force(
        &self,
        nondimensional_extension: Scalar,
        // ) -> Result<Scalar, SingleChainError> {
    ) -> Scalar {
        // SingleChainThermodynamics::nondimensional_force(&self.single_chain_model, nondimensional_extension)
        SingleChainThermodynamics::nondimensional_force(
            &self.single_chain_model,
            nondimensional_extension,
        )
        .unwrap()
    }
    /// ???
    fn nondimensional_stiffness(
        &self,
        nondimensional_extension: Scalar,
        // ) -> Result<Scalar, SingleChainError> {
    ) -> Scalar {
        // SingleChainThermodynamics::nondimensional_force(&self.single_chain_model, nondimensional_extension)
        SingleChainThermodynamics::nondimensional_stiffness(
            &self.single_chain_model,
            nondimensional_extension,
        )
        .unwrap()
    }
    /// Returns the number of links.
    pub fn number_of_links(&self) -> Scalar {
        self.single_chain_model.number_of_links() as Scalar
    }
    /// Returns the single-chain model.
    pub fn single_chain_model(&self) -> &T {
        &self.single_chain_model
    }
}

impl<T> Solid for EightChain<T>
where
    T: SingleChainThermodynamics,
{
    fn bulk_modulus(&self) -> Scalar {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Scalar {
        self.shear_modulus
    }
}

impl<T> Elastic for EightChain<T>
where
    T: SingleChainThermodynamics,
{
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let (
            deviatoric_isochoric_left_cauchy_green_deformation,
            isochoric_left_cauchy_green_deformation_trace,
        ) = (deformation_gradient.left_cauchy_green() / jacobian.powf(TWO_THIRDS))
            .deviatoric_and_trace();
        let gamma =
            (isochoric_left_cauchy_green_deformation_trace / 3.0 / self.number_of_links()).sqrt();
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
        let (
            deviatoric_isochoric_left_cauchy_green_deformation,
            isochoric_left_cauchy_green_deformation_trace,
        ) = (left_cauchy_green_deformation / jacobian.powf(TWO_THIRDS)).deviatoric_and_trace();
        let gamma =
            (isochoric_left_cauchy_green_deformation_trace / 3.0 / self.number_of_links()).sqrt();
        let gamma_0 = (1.0 / self.number_of_links()).sqrt();
        let eta = self.nondimensional_force(gamma);
        let scaled_shear_modulus =
            gamma_0 / self.nondimensional_force(gamma_0) * self.shear_modulus() * eta
                / gamma
                / jacobian.powf(FIVE_THIRDS);
        let scaled_deviatoric_isochoric_left_cauchy_green_deformation =
            deviatoric_left_cauchy_green_deformation * scaled_shear_modulus;
        let term = CauchyTangentStiffness::dyad_ij_kl(
            &scaled_deviatoric_isochoric_left_cauchy_green_deformation,
            &(deviatoric_isochoric_left_cauchy_green_deformation
                * &inverse_transpose_deformation_gradient
                * ((self.nondimensional_stiffness(gamma) / eta - 1.0 / gamma)
                    / 3.0
                    / self.number_of_links()
                    / gamma)),
        );
        //
        // Need to replace above with something that uses nondimensional_stiffness()
        //
        Ok(
            (CauchyTangentStiffness::dyad_ik_jl(&IDENTITY, deformation_gradient)
                + CauchyTangentStiffness::dyad_il_jk(deformation_gradient, &IDENTITY)
                - CauchyTangentStiffness::dyad_ij_kl(&IDENTITY, deformation_gradient)
                    * (TWO_THIRDS))
                * scaled_shear_modulus
                + CauchyTangentStiffness::dyad_ij_kl(
                    &(IDENTITY * (0.5 * self.bulk_modulus() * (jacobian + 1.0 / jacobian))
                        - scaled_deviatoric_isochoric_left_cauchy_green_deformation
                            * (FIVE_THIRDS)),
                    &inverse_transpose_deformation_gradient,
                )
                + term,
        )
    }
}

impl<T> Hyperelastic for EightChain<T>
where
    T: SingleChainThermodynamics,
{
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let isochoric_left_cauchy_green_deformation =
            deformation_gradient.left_cauchy_green() / jacobian.powf(TWO_THIRDS);
        let gamma =
            (isochoric_left_cauchy_green_deformation.trace() / 3.0 / self.number_of_links()).sqrt();
        let eta = self.nondimensional_force(gamma);
        let gamma_0 = (1.0 / self.number_of_links()).sqrt();
        let eta_0 = self.nondimensional_force(gamma_0);
        //
        // If end up re-using so much of ArrudaBoyce should make helper function to share.
        //
        Ok(3.0 * gamma_0 / eta_0
            * self.shear_modulus()
            * self.number_of_links()
            * (gamma * eta - gamma_0 * eta_0 - (eta_0 * eta.sinh() / (eta * eta_0.sinh())).ln())
            + 0.5 * self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln()))
    }
}
