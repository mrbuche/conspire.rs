#[cfg(test)]
mod test;

use crate::{
    constitutive::{ConstitutiveError, chemical::Chemical, diffusion::Diffusion},
    math::{IDENTITY_00, Quantity, Tensor},
    mechanics::{ConcentrationGradient, MolarFlux, MolarFluxTangent},
    units::{Diffusivity, MolarFlowRate, MolarFlowRateConcentrationDensity},
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct Fick {
    /// The diffusivity $`D`$.
    pub diffusivity: Quantity<Diffusivity>,
}

impl Chemical for Fick {}

impl Diffusion for Fick {
    type Balance = MolarFlowRate;
    #[doc = include_str!("potential.md")]
    fn potential(
        &self,
        concentration_gradient: &ConcentrationGradient,
    ) -> Result<Quantity<MolarFlowRateConcentrationDensity>, ConstitutiveError> {
        Ok(concentration_gradient.norm_squared() * self.diffusivity * 0.5)
    }
    #[doc = include_str!("molar_flux.md")]
    fn flux(
        &self,
        concentration_gradient: &ConcentrationGradient,
    ) -> Result<MolarFlux, ConstitutiveError> {
        Ok(concentration_gradient * -self.diffusivity)
    }
    #[doc = include_str!("molar_flux_tangent.md")]
    fn tangent(
        &self,
        _concentration_gradient: &ConcentrationGradient,
    ) -> Result<MolarFluxTangent, ConstitutiveError> {
        Ok(IDENTITY_00 * -self.diffusivity)
    }
}
