//! Thermal conduction constitutive models.

#[cfg(test)]
pub mod test;

mod fourier;

use super::{Constitutive, HeatFlux, Parameters, Scalar, TemperatureGradient, Thermal};

pub use fourier::Fourier;

/// Required methods for thermal conduction constitutive models.
pub trait ThermalConduction<P>
where
    Self: Constitutive<P> + Thermal<P>,
{
    /// Calculates and returns the heat flux.
    fn heat_flux(&self, temperature_gradient: &TemperatureGradient) -> HeatFlux;
}
