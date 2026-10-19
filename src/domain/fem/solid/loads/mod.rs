#[cfg(test)]
mod test;

use crate::{
    domain::{
        boundary::{ConservativeAt, LoadAt, Loads, PowerAt},
        multiphysics::solid_thermal::{NodalForcesSolidThermal, NodalStatesSolidThermal},
        solid::loads::{DeadLoad, SolidLayout},
    },
    fem::{
        NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::element::{
            ElementNodalReferenceCoordinates,
            surface::{
                SurfaceFiniteElement,
                linear::{Quadrilateral, Triangle},
            },
        },
        solid::NodalForcesSolid,
    },
    geometry::mesh::SideSet,
    math::Quantity,
    math::Tensor,
    mechanics::Traction,
    units::{Area, Energy, Power, Time},
};

/// The area that each node has of the faces of a side set.
///
/// Fails if the side set has no faces, or a face that is not a triangle or a quadrilateral.
pub fn nodal_areas(
    coordinates: &NodalReferenceCoordinates<3>,
    side_set: &SideSet,
) -> Result<Vec<Quantity<Area>>, String> {
    if side_set.faces().len() != side_set.sides().len() {
        return Err("The side set has no faces on the mesh.".to_string());
    }
    let mut areas = vec![Quantity::<Area>::new(0.0); coordinates.len()];
    side_set.faces().iter().try_for_each(|face| {
        let nodal_areas = match *face.as_slice() {
            [a, b, c] => Triangle::nodal_areas(&ElementNodalReferenceCoordinates::from([
                coordinates[a].clone(),
                coordinates[b].clone(),
                coordinates[c].clone(),
            ])),
            [a, b, c, d] => Quadrilateral::nodal_areas(&ElementNodalReferenceCoordinates::from([
                coordinates[a].clone(),
                coordinates[b].clone(),
                coordinates[c].clone(),
                coordinates[d].clone(),
            ])),
            _ => {
                return Err(format!(
                    "A face with {} nodes is not a triangle or a quadrilateral.",
                    face.len()
                ));
            }
        };
        face.iter()
            .zip(nodal_areas)
            .for_each(|(&node, area)| areas[node] += area);
        Ok(())
    })?;
    Ok(areas)
}

/// The nodal forces of a dead traction per unit reference area on the faces of a side set.
///
/// Fails if the side set has no faces, or a face that is not a triangle or a quadrilateral.
pub fn traction_forces(
    coordinates: &NodalReferenceCoordinates<3>,
    side_set: &SideSet,
    traction: &Traction,
) -> Result<NodalForcesSolid<3>, String> {
    Ok(nodal_areas(coordinates, side_set)?
        .into_iter()
        .map(|area| traction * area)
        .collect())
}

impl<L, T> Loads<L, T>
where
    L: SolidLayout,
{
    /// Adds a dead traction per unit reference area on the faces of a side set.
    pub fn traction(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        traction: &Traction,
    ) -> Result<Loads<L, (T, DeadLoad)>, String> {
        Ok(self.with(DeadLoad::new(
            coordinates,
            traction_forces(coordinates, side_set, traction)?,
        )?))
    }
}

/// A traction per unit reference area on the faces of a side set, whose force does not change
/// as the body deforms, and is a function of time.
pub struct TimedTraction<G> {
    areas: Vec<Quantity<Area>>,
    reference: NodalCoordinates<3>,
    traction: G,
}

type WithTimedTraction<L, T, G> = Result<Loads<L, (T, TimedTraction<G>)>, String>;

impl<L, T> Loads<L, T>
where
    L: SolidLayout,
{
    /// Adds a dead traction per unit reference area on the faces of a side set, that is a
    /// function of time.
    pub fn traction_with(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        traction: impl Fn(Quantity<Time>) -> Traction,
    ) -> WithTimedTraction<L, T, impl Fn(Quantity<Time>) -> Traction> {
        Ok(self.with(TimedTraction {
            areas: nodal_areas(coordinates, side_set)?,
            reference: coordinates.clone().into(),
            traction,
        }))
    }
}

impl<G> TimedTraction<G>
where
    G: Fn(Quantity<Time>) -> Traction,
{
    fn forces(&self, time: Quantity<Time>, nodes: usize) -> Result<NodalForcesSolid<3>, String> {
        if nodes != self.areas.len() {
            return Err("The traction is on a mesh with another number of nodes.".to_string());
        }
        let traction = (self.traction)(time);
        Ok(self.areas.iter().map(|&area| &traction * area).collect())
    }
}

impl<G> LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>> for TimedTraction<G>
where
    G: Fn(Quantity<Time>) -> Traction,
{
    fn subtract_at(
        &self,
        time: Quantity<Time>,
        nodal_coordinates: &NodalCoordinates<3>,
        residual: &mut NodalForcesSolid<3>,
    ) -> Result<(), String> {
        residual
            .iter_mut()
            .zip(self.forces(time, nodal_coordinates.len())?)
            .for_each(|(residual, force)| *residual -= force);
        Ok(())
    }
}

impl<G> ConservativeAt<NodalCoordinates<3>, Quantity<Energy>> for TimedTraction<G>
where
    G: Fn(Quantity<Time>) -> Traction,
{
    fn potential_energy_at(
        &self,
        time: Quantity<Time>,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Energy>, String> {
        Ok(-nodal_coordinates
            .iter()
            .zip(&self.reference)
            .zip(self.forces(time, nodal_coordinates.len())?)
            .map(|((coordinate, reference), force)| force * &(coordinate - reference))
            .sum::<Quantity<Energy>>())
    }
}

impl<G> PowerAt<NodalCoordinates<3>, NodalVelocities<3>, Quantity<Power>> for TimedTraction<G>
where
    G: Fn(Quantity<Time>) -> Traction,
{
    fn power_at(
        &self,
        time: Quantity<Time>,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Result<Quantity<Power>, String> {
        Ok(-nodal_velocities
            .iter()
            .zip(self.forces(time, nodal_coordinates.len())?)
            .map(|(velocity, force)| force * velocity)
            .sum::<Quantity<Power>>())
    }
}

impl<G> LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal> for TimedTraction<G>
where
    G: Fn(Quantity<Time>) -> Traction,
{
    fn subtract_at(
        &self,
        time: Quantity<Time>,
        nodal_states: &NodalStatesSolidThermal,
        residual: &mut NodalForcesSolidThermal,
    ) -> Result<(), String> {
        residual
            .iter_mut()
            .zip(self.forces(time, nodal_states.len())?)
            .for_each(|(residual, force)| residual.0 -= force);
        Ok(())
    }
}
