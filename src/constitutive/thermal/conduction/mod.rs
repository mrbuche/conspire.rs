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
    constitutive::{ConstitutiveError, diffusion::Diffusion, thermal::Thermal},
    mechanics::{HeatFlux, HeatFluxTangent, TemperatureGradient},
    units::Temperature,
};

pub use fourier::Fourier;

/// Required methods for thermal conduction constitutive models.
pub trait ThermalConduction
where
    Self: Thermal + Diffusion<Field = Temperature>,
{
    /// Calculates and returns the heat flux.
    fn heat_flux(
        &self,
        temperature_gradient: &TemperatureGradient,
    ) -> Result<HeatFlux, ConstitutiveError> {
        self.flux(temperature_gradient)
    }
    /// Calculates and returns the tangent to the heat flux.
    fn heat_flux_tangent(
        &self,
        temperature_gradient: &TemperatureGradient,
    ) -> Result<HeatFluxTangent, ConstitutiveError> {
        self.tangent(temperature_gradient)
    }
}

impl<C> ThermalConduction for C where C: Thermal + Diffusion<Field = Temperature> {}
