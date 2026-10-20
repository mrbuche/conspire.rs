#[cfg(test)]
mod test;

use crate::{
    constitutive::diffusion::Diffusive,
    domain::{
        boundary::{Field, Loads, SolidThermal},
        diffusion::loads::{Convection, NodalSource, TimedSource},
    },
    fem::{NodalReferenceCoordinates, solid::loads::nodal_areas},
    geometry::mesh::SideSet,
    math::Quantity,
    units::{Area, Power, PowerPerArea, PowerPerAreaTemperature, Temperature, Time},
};
use std::ops::Mul;

/// A flux per unit reference area.
pub type TimedFlux<G> = TimedSource<Area, G>;

type Flux<L> = Quantity<<L as Diffusive>::Flux>;
type WithSource<U, T, L> = Loads<Field<U>, (T, NodalSource<L>)>;
type WithTimedFlux<U, T, G> = Result<Loads<Field<U>, (T, TimedFlux<G>)>, String>;
type WithHeat<T> = Loads<SolidThermal<3>, (T, NodalSource<Power>)>;
type WithTimedHeat<T, G> = Result<Loads<SolidThermal<3>, (T, TimedFlux<G>)>, String>;

type WithConvection<U, T, L> = Result<Loads<Field<U>, (T, Convection<L>)>, String>;
type WithHeatConvection<T> = Result<Loads<SolidThermal<3>, (T, Convection<Power>)>, String>;

impl<U, T> Loads<Field<U>, T> {
    /// Adds a flux per unit reference area through the faces of a side set that is the
    /// coefficient times the difference of the field from the ambient field, such as convection.
    pub fn convection_through<L>(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        coefficient: Quantity<L::TangentPerLength>,
        ambient: Quantity<L::Field>,
    ) -> WithConvection<U, T, L>
    where
        L: Diffusive<Field = U>,
        Quantity<L::TangentPerLength>: Mul<Quantity<Area>, Output = Quantity<L::Stiffness>>,
    {
        let coefficients = nodal_areas(coordinates, side_set)?
            .into_iter()
            .map(|area| coefficient * area)
            .collect();
        Ok(self.convection(ambient, coefficients))
    }
    /// Adds a flux per unit reference area through the faces of a side set, such as heating.
    pub fn flux<L>(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        flux: Flux<L>,
    ) -> Result<WithSource<U, T, L>, String>
    where
        L: Diffusive<Field = U>,
        Quantity<L::Flux>: Mul<Quantity<Area>, Output = Quantity<L>>,
    {
        let sources = nodal_areas(coordinates, side_set)?
            .into_iter()
            .map(|area| flux * area)
            .collect();
        Ok(self.source(sources))
    }
    /// Adds a flux per unit reference area through the side set faces, that is a function of time.
    pub fn flux_with<L, G>(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        flux: G,
    ) -> WithTimedFlux<U, T, G>
    where
        L: Diffusive<Field = U>,
        G: Fn(Quantity<Time>) -> Flux<L>,
    {
        Ok(self.with(TimedSource::new(nodal_areas(coordinates, side_set)?, flux)))
    }
}

impl<T> Loads<SolidThermal<3>, T> {
    /// Adds a heat flux per unit reference area through the faces of a side set.
    pub fn flux(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        flux: Quantity<PowerPerArea>,
    ) -> Result<WithHeat<T>, String> {
        let sources = nodal_areas(coordinates, side_set)?
            .into_iter()
            .map(|area| flux * area)
            .collect();
        Ok(self.source(sources))
    }
    /// Adds a heat flux per unit reference area through the faces of a side set, that is a
    /// function of time.
    pub fn flux_with<G>(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        flux: G,
    ) -> WithTimedHeat<T, G>
    where
        G: Fn(Quantity<Time>) -> Quantity<PowerPerArea>,
    {
        Ok(self.with(TimedSource::new(nodal_areas(coordinates, side_set)?, flux)))
    }
}

impl<T> Loads<SolidThermal<3>, T> {
    /// Adds a heat flux per unit reference area through the faces of a side set that is the
    /// coefficient times the difference of the temperature from the ambient temperature.
    pub fn convection_through(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        coefficient: Quantity<PowerPerAreaTemperature>,
        ambient: Quantity<Temperature>,
    ) -> WithHeatConvection<T> {
        let coefficients = nodal_areas(coordinates, side_set)?
            .into_iter()
            .map(|area| coefficient * area)
            .collect();
        Ok(self.convection(ambient, coefficients))
    }
}
