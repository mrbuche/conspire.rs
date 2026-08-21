#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        thermal::{Thermal, conduction::ThermalConduction},
    },
    math::{ContractWith, IDENTITY_00, Quantity},
    mechanics::{HeatFlux, HeatFluxTangent, TemperatureGradient},
    units::{PowerPerLengthTemperature, PowerTemperatureDensity},
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

impl ThermalConduction for Fourier {
    #[doc = include_str!("potential.md")]
    fn potential(
        &self,
        temperature_gradient: &TemperatureGradient,
    ) -> Result<Quantity<PowerTemperatureDensity>, ConstitutiveError> {
        // A temperature gradient squared names nothing, so the potential is
        // stated as the flux it gives contracted with the gradient.
        Ok(
            (temperature_gradient * self.thermal_conductivity())
                .contract_with(temperature_gradient)
                * 0.5,
        )
    }
    #[doc = include_str!("heat_flux.md")]
    fn heat_flux(
        &self,
        temperature_gradient: &TemperatureGradient,
    ) -> Result<HeatFlux, ConstitutiveError> {
        Ok(temperature_gradient * -self.thermal_conductivity())
    }
    #[doc = include_str!("heat_flux_tangent.md")]
    fn heat_flux_tangent(
        &self,
        _temperature_gradient: &TemperatureGradient,
    ) -> Result<HeatFluxTangent, ConstitutiveError> {
        Ok(IDENTITY_00 * -self.thermal_conductivity())
    }
}
