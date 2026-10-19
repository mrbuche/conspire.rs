#[cfg(test)]
mod test;

use crate::{
    constitutive::diffusion::Diffusive,
    domain::{
        boundary::{ConservativeAt, Field, LoadAt, Loads, SolidThermal},
        diffusion::loads::NodalSource,
        multiphysics::solid_thermal::{NodalForcesSolidThermal, NodalStatesSolidThermal},
    },
    fem::{
        NodalReferenceCoordinates,
        diffusion::{NodalFields, NodalForces},
        solid::loads::nodal_areas,
    },
    geometry::mesh::SideSet,
    math::{Quantity, Tensor},
    units::{Area, Power, PowerPerArea, Time},
};
use std::{marker::PhantomData, ops::Mul};

/// A flux per unit reference area through the faces of a side set that is a function of time.
pub struct TimedFlux<L, G> {
    areas: Vec<Quantity<Area>>,
    flux: G,
    unit: PhantomData<L>,
}

type WithTimedFlux<U, T, L, G> = Result<Loads<Field<U>, (T, TimedFlux<L, G>)>, String>;
type WithSource<U, T, L> = Loads<Field<U>, (T, NodalSource<L>)>;
type Flux<L> = Quantity<<L as Diffusive>::Flux>;

impl<U, T> Loads<Field<U>, T> {
    /// Adds a flux per unit reference area through the faces of a side set, such as heating.
    pub fn flux<L>(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        flux: Quantity<L::Flux>,
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
    pub fn flux_with<L>(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        flux: impl Fn(Quantity<Time>) -> Flux<L>,
    ) -> WithTimedFlux<U, T, L, impl Fn(Quantity<Time>) -> Flux<L>>
    where
        L: Diffusive<Field = U>,
    {
        Ok(self.with(TimedFlux {
            areas: nodal_areas(coordinates, side_set)?,
            flux,
            unit: PhantomData,
        }))
    }
}

impl<T> Loads<SolidThermal<3>, T> {
    /// Adds a heat flux per unit reference area through the faces of a side set.
    pub fn flux(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        flux: Quantity<PowerPerArea>,
    ) -> Result<Loads<SolidThermal<3>, (T, NodalSource<Power>)>, String> {
        let sources = nodal_areas(coordinates, side_set)?
            .into_iter()
            .map(|area| flux * area)
            .collect();
        Ok(self.source(sources))
    }
}

type WithTimedHeat<T, G> = Result<Loads<SolidThermal<3>, (T, TimedFlux<Power, G>)>, String>;

impl<T> Loads<SolidThermal<3>, T> {
    /// Adds a heat flux per unit reference area through the faces of a side set, that is a
    /// function of time.
    pub fn flux_with(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        flux: impl Fn(Quantity<Time>) -> Quantity<PowerPerArea>,
    ) -> WithTimedHeat<T, impl Fn(Quantity<Time>) -> Quantity<PowerPerArea>> {
        Ok(self.with(TimedFlux {
            areas: nodal_areas(coordinates, side_set)?,
            flux,
            unit: PhantomData,
        }))
    }
}

impl<G> LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal> for TimedFlux<Power, G>
where
    G: Fn(Quantity<Time>) -> Quantity<PowerPerArea>,
{
    fn subtract_at(
        &self,
        time: Quantity<Time>,
        nodal_states: &NodalStatesSolidThermal,
        residual: &mut NodalForcesSolidThermal,
    ) -> Result<(), String> {
        if nodal_states.len() != self.areas.len() {
            return Err("The flux is on a mesh with another number of nodes.".to_string());
        }
        let flux = (self.flux)(time);
        residual
            .iter_mut()
            .zip(&self.areas)
            .for_each(|(residual, &area)| residual.1 -= flux * area);
        Ok(())
    }
}

impl<L, G> LoadAt<NodalFields<L>, NodalForces<L>> for TimedFlux<L, G>
where
    L: Diffusive,
    G: Fn(Quantity<Time>) -> Quantity<L::Flux>,
    Quantity<L::Flux>: Mul<Quantity<Area>, Output = Quantity<L>>,
{
    fn subtract_at(
        &self,
        time: Quantity<Time>,
        nodal_fields: &NodalFields<L>,
        residual: &mut NodalForces<L>,
    ) -> Result<(), String> {
        if nodal_fields.len() != self.areas.len() {
            return Err("The flux is on a mesh with another number of nodes.".to_string());
        }
        let flux = (self.flux)(time);
        residual
            .iter_mut()
            .zip(&self.areas)
            .for_each(|(residual, &area)| *residual -= flux * area);
        Ok(())
    }
}

impl<L, G> ConservativeAt<NodalFields<L>, Quantity<L::Energy>> for TimedFlux<L, G>
where
    L: Diffusive,
    G: Fn(Quantity<Time>) -> Flux<L>,
    Quantity<L::Flux>: Mul<Quantity<Area>, Output = Quantity<L>>,
    Quantity<L::Field>: Mul<Quantity<L>, Output = Quantity<L::Energy>>,
{
    fn potential_energy_at(
        &self,
        time: Quantity<Time>,
        nodal_fields: &NodalFields<L>,
    ) -> Result<Quantity<L::Energy>, String> {
        if nodal_fields.len() != self.areas.len() {
            return Err("The flux is on a mesh with another number of nodes.".to_string());
        }
        let flux = (self.flux)(time);
        Ok(-nodal_fields
            .iter()
            .zip(&self.areas)
            .map(|(&field, &area)| field * (flux * area))
            .sum::<Quantity<L::Energy>>())
    }
}
