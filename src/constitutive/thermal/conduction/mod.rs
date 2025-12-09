//! Thermal conduction constitutive models.

#[cfg(test)]
pub mod test;

mod fourier;

use super::{super::ConstitutiveError, HeatFlux, Scalar, TemperatureGradient, Thermal};

pub use fourier::Fourier;

/// Required methods for thermal conduction constitutive models.
pub trait ThermalConduction
where
    Self: Thermal,
{
    /// Calculates and returns the heat flux.
    fn heat_flux(
        &self,
        temperature_gradient: &TemperatureGradient,
    ) -> Result<HeatFlux, ConstitutiveError>;
}
