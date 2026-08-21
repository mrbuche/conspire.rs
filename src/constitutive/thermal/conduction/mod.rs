//! Thermal conduction constitutive models.
//!
//! ---
//!
#![doc = include_str!("doc.md")]

#[cfg(feature = "doc")]
pub mod doc;

#[cfg(test)]
pub mod test;

mod fourier;

use crate::{
    constitutive::{ConstitutiveError, thermal::Thermal},
    math::Quantity,
    mechanics::{HeatFlux, HeatFluxTangent, TemperatureGradient},
    units::PowerTemperatureDensity,
};

pub use fourier::Fourier;

/// Required methods for thermal conduction constitutive models.
pub trait ThermalConduction
where
    Self: Thermal,
{
    /// Calculates and returns the potential.
    fn potential(
        &self,
        temperature_gradient: &TemperatureGradient,
    ) -> Result<Quantity<PowerTemperatureDensity>, ConstitutiveError>;
    /// Calculates and returns the heat flux.
    fn heat_flux(
        &self,
        temperature_gradient: &TemperatureGradient,
    ) -> Result<HeatFlux, ConstitutiveError>;
    /// Calculates and returns the tangent to the heat flux.
    fn heat_flux_tangent(
        &self,
        temperature_gradient: &TemperatureGradient,
    ) -> Result<HeatFluxTangent, ConstitutiveError>;
}
