#[cfg(test)]
mod test;

use crate::{
    constitutive::{ConstitutiveError, diffusion::Diffusion, thermal::Thermal},
    math::{IDENTITY_00, Quantity, Tensor},
    mechanics::{HeatFlux, HeatFluxTangent, TemperatureGradient},
    units::{Power, PowerPerLengthTemperature, PowerTemperatureDensity},
};

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct Fourier {
    /// The thermal conductivity $`k`$.
    pub thermal_conductivity: Quantity<PowerPerLengthTemperature>,
}

impl Fourier {
    fn thermal_conductivity(&self) -> Quantity<PowerPerLengthTemperature> {
        self.thermal_conductivity
    }
}

impl Thermal for Fourier {}

impl Diffusion for Fourier {
    type Balance = Power;
    #[doc = include_str!("potential.md")]
    fn potential(
        &self,
        temperature_gradient: &TemperatureGradient,
    ) -> Result<Quantity<PowerTemperatureDensity>, ConstitutiveError> {
        Ok(temperature_gradient.norm_squared() * self.thermal_conductivity() * 0.5)
    }
    #[doc = include_str!("heat_flux.md")]
    fn flux(
        &self,
        temperature_gradient: &TemperatureGradient,
    ) -> Result<HeatFlux, ConstitutiveError> {
        Ok(temperature_gradient * -self.thermal_conductivity())
    }
    #[doc = include_str!("heat_flux_tangent.md")]
    fn tangent(
        &self,
        _temperature_gradient: &TemperatureGradient,
    ) -> Result<HeatFluxTangent, ConstitutiveError> {
        Ok(IDENTITY_00 * -self.thermal_conductivity())
    }
}
