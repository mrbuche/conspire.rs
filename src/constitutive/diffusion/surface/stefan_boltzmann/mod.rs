#[cfg(test)]
mod test;

use super::{SurfaceField, SurfaceFlux, SurfaceFluxPerArea, SurfacePotential, SurfaceTangent};
use crate::{
    math::{Quantity, Scalar},
    units::{
        Area, Power, PowerTemperature, STEFAN_BOLTZMANN_CONSTANT, Temperature, TemperatureFourth,
    },
};

/// The Stefan-Boltzmann law of the radiation of heat from a gray surface to its surroundings,
/// where the flux is the Stefan-Boltzmann constant times the emissivity times the difference of
/// the fourth powers of the temperature and of the ambient temperature.
///
/// The temperatures are absolute.
#[derive(Clone, Debug)]
pub struct StefanBoltzmann {
    /// The emissivity of the surface.
    pub emissivity: Scalar,
    /// The temperature of the surroundings.
    pub ambient: Quantity<Temperature>,
}

fn fourth(temperature: Quantity<Temperature>) -> Quantity<TemperatureFourth> {
    let squared = temperature * temperature;
    squared * squared
}

impl SurfaceFlux for StefanBoltzmann {
    type Balance = Power;
    fn flux(&self, field: SurfaceField<Self>) -> SurfaceFluxPerArea<Self> {
        STEFAN_BOLTZMANN_CONSTANT * self.emissivity * (fourth(field) - fourth(self.ambient))
    }
    fn tangent(&self, field: SurfaceField<Self>) -> SurfaceTangent<Self> {
        STEFAN_BOLTZMANN_CONSTANT * self.emissivity * (field * field * field) * 4.0
    }
}

impl SurfacePotential for StefanBoltzmann {
    fn potential(
        &self,
        field: SurfaceField<Self>,
        area: Quantity<Area>,
    ) -> Quantity<PowerTemperature> {
        STEFAN_BOLTZMANN_CONSTANT
            * self.emissivity
            * area
            * (fourth(field) * field * 0.2 - fourth(self.ambient) * field)
    }
}
