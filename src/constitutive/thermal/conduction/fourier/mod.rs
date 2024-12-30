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
pub struct Fourier<'a> {
    parameters: Parameters<'a>,
}

impl Fourier<'_> {
    fn get_thermal_conductivity(&self) -> &Scalar {
        &self.parameters[0]
    }
}

impl<'a> Constitutive<'a> for Fourier<'a> {
    fn new(parameters: Parameters<'a>) -> Self {
        Self { parameters }
    }
}
impl<'a> Thermal<'a> for Fourier<'a> {}

impl<'a> ThermalConduction<'a> for Fourier<'a> {
    /// Calculates and returns the heat flux.
    ///
    /// ```math
    /// \mathbf{q}(\nabla T) = -k\nabla T
    /// ```
    fn calculate_heat_flux(&self, temperature_gradient: &TemperatureGradient) -> HeatFlux {
        temperature_gradient * -self.get_thermal_conductivity()
    }
}
