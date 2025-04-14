#[cfg(test)]
mod test;

use super::{
    Constitutive, HeatFlux, Parameters, Scalar, TemperatureGradient, Thermal, ThermalConduction,
};

/// The Fourier thermal conduction constitutive model.
///
/// **Parameters**
/// - The thermal conductivity $`k`$.
///
/// **External variables**
/// - The temperature gradient $`\nabla T`$.
///
/// **Internal variables**
/// - None.
#[derive(Debug)]
pub struct Fourier<P> {
    parameters: P,
}

impl<P> Fourier<P> where P: Parameters {
    fn thermal_conductivity(&self) -> &Scalar {
        self.parameters.get(0)
    }
}

impl<P> Constitutive<P> for Fourier<P> where P: Parameters {
    fn new(parameters: P) -> Self {
        Self { parameters }
    }
}
impl<P> Thermal<P> for Fourier<P> where P: Parameters {}

impl<P> ThermalConduction<P> for Fourier<P> where P: Parameters {
    /// Calculates and returns the heat flux.
    ///
    /// ```math
    /// \mathbf{q}(\nabla T) = -k\nabla T
    /// ```
    fn heat_flux(&self, temperature_gradient: &TemperatureGradient) -> HeatFlux {
        temperature_gradient * -self.thermal_conductivity()
    }
}
