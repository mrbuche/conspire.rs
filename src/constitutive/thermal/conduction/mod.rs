//! Thermal conduction constitutive models.

#[cfg(test)]
pub mod test;

mod fourier;

use crate::{
    constitutive::{ConstitutiveError, thermal::Thermal},
    mechanics::{HeatFlux, HeatFluxTangent, Scalar, TemperatureGradient},
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
    ) -> Result<Scalar, ConstitutiveError>;
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
