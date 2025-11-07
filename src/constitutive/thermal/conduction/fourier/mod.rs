#[cfg(test)]
mod test;

use super::{HeatFlux, Scalar, TemperatureGradient, Thermal, ThermalConduction};

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
pub struct Fourier {
    /// The thermal conductivity $`k`$.
    pub thermal_conductivity: Scalar,
}

impl Fourier {
    fn thermal_conductivity(&self) -> Scalar {
        self.thermal_conductivity
    }
}

impl Thermal for Fourier {}

impl ThermalConduction for Fourier {
    /// Calculates and returns the heat flux.
    ///
    /// ```math
    /// \mathbf{q}(\nabla T) = -k\nabla T
    /// ```
    fn heat_flux(&self, temperature_gradient: &TemperatureGradient) -> HeatFlux {
        temperature_gradient * -self.thermal_conductivity()
    }
}
