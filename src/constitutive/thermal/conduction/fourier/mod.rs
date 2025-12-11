#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        thermal::{Thermal, conduction::ThermalConduction},
    },
    math::IDENTITY_00,
    mechanics::{HeatFlux, HeatFluxTangent, Scalar, TemperatureGradient},
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
#[derive(Clone, Debug)]
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
    /// Calculates and returns the potential.
    ///
    /// ```math
    /// a(\nabla T) = \frac{1}{2}k\nabla T\cdot\nabla T
    /// ```
    fn potential(
        &self,
        temperature_gradient: &TemperatureGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        Ok(0.5 * self.thermal_conductivity() * (temperature_gradient * temperature_gradient))
    }
    /// Calculates and returns the heat flux.
    ///
    /// ```math
    /// \mathbf{q}(\nabla T) = -k\nabla T
    /// ```
    fn heat_flux(
        &self,
        temperature_gradient: &TemperatureGradient,
    ) -> Result<HeatFlux, ConstitutiveError> {
        Ok(temperature_gradient * -self.thermal_conductivity())
    }
    /// Calculates and returns the tangent to the heat flux.
    ///
    /// ```math
    /// \frac{\partial\mathbf{q}}{\partial\nabla T} = -k\mathbf{I}
    /// ```
    fn heat_flux_tangent(
        &self,
        _temperature_gradient: &TemperatureGradient,
    ) -> Result<HeatFluxTangent, ConstitutiveError> {
        Ok(IDENTITY_00 * -self.thermal_conductivity())
    }
}
