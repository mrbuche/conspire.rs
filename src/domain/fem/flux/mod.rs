#[cfg(test)]
mod test;

use crate::{
    constitutive::diffusion::{Diffusive, surface::SurfaceFlux},
    domain::{
        boundary::{Field, Loads, SolidThermal},
        diffusion::loads::{NodalSource, Surface, TimedSource, TimedSurface},
    },
    fem::{NodalReferenceCoordinates, solid::loads::nodal_areas},
    geometry::mesh::SideSet,
    math::Quantity,
    units::{Area, Power, PowerPerArea, Time},
};
use std::ops::Mul;

/// A flux per unit reference area.
pub type TimedFlux<G> = TimedSource<Area, G>;

type Balance<S> = <S as SurfaceFlux>::Balance;
type Flux<L> = Quantity<<L as Diffusive>::Flux>;
type WithSource<U, T, L> = Loads<Field<U>, (T, NodalSource<L>)>;
type WithTimedFlux<U, T, G> = Result<Loads<Field<U>, (T, TimedFlux<G>)>, String>;
type WithHeat<T> = Loads<SolidThermal<3>, (T, NodalSource<Power>)>;
type WithTimedHeat<T, G> = Result<Loads<SolidThermal<3>, (T, TimedFlux<G>)>, String>;

type WithSurface<U, T, S> = Result<Loads<Field<U>, (T, Surface<S>)>, String>;
type WithTimedSurface<U, T, G> = Result<Loads<Field<U>, (T, TimedSurface<G>)>, String>;
type WithTimedHeatSurface<T, G> = Result<Loads<SolidThermal<3>, (T, TimedSurface<G>)>, String>;
type WithHeatSurface<T, S> = Result<Loads<SolidThermal<3>, (T, Surface<S>)>, String>;

impl<U, T> Loads<Field<U>, T> {
    /// Adds a flux through the faces of a side set that is a model of the field on them, such as
    /// convection.
    pub fn surface<S>(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        model: S,
    ) -> WithSurface<U, T, S>
    where
        S: SurfaceFlux,
        Balance<S>: Diffusive<Field = U>,
    {
        Ok(self.with_surface(nodal_areas(coordinates, side_set)?, model))
    }
    /// Adds a flux through the faces of a side set that is a model of the field on them, where the
    /// model is a function of time.
    pub fn surface_with<S, G>(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        model: G,
    ) -> WithTimedSurface<U, T, G>
    where
        S: SurfaceFlux,
        Balance<S>: Diffusive<Field = U>,
        G: Fn(Quantity<Time>) -> S,
    {
        Ok(self.with_timed_surface(nodal_areas(coordinates, side_set)?, model))
    }
}

impl<U, T> Loads<Field<U>, T> {
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
    /// Adds a heat flux through the faces of a side set that is a model of the temperature on
    /// them, such as convection.
    pub fn surface<S>(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        model: S,
    ) -> WithHeatSurface<T, S>
    where
        S: SurfaceFlux<Balance = Power>,
    {
        Ok(self.with_surface(nodal_areas(coordinates, side_set)?, model))
    }
    /// Adds a heat flux through the faces of a side set that is a model of the temperature on
    /// them, where the model is a function of time.
    pub fn surface_with<S, G>(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        model: G,
    ) -> WithTimedHeatSurface<T, G>
    where
        S: SurfaceFlux<Balance = Power>,
        G: Fn(Quantity<Time>) -> S,
    {
        Ok(self.with_timed_surface(nodal_areas(coordinates, side_set)?, model))
    }
}
