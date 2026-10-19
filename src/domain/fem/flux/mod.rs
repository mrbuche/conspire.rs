#[cfg(test)]
mod test;

use crate::{
    constitutive::diffusion::Diffusive,
    domain::{
        boundary::{Field, LoadAt, Loads},
        diffusion::loads::NodalSource,
    },
    fem::{
        NodalReferenceCoordinates,
        diffusion::{NodalFields, NodalForces},
        solid::loads::nodal_areas,
    },
    geometry::mesh::SideSet,
    math::{Quantity, Tensor},
    units::{Area, Time},
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
