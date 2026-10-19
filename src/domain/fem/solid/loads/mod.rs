#[cfg(test)]
mod test;

use crate::{
    domain::{
        boundary::{ConservativeAt, LoadAt, Loads, PowerAt, Solid},
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
    mechanics::{Force, Traction},
    units::{Energy, Power, Time},
};

/// The nodal forces of a dead traction per unit reference area on the faces of a side set.
///
/// Fails if the side set has no faces, or a face that is not a triangle or a quadrilateral.
pub fn traction_forces(
    coordinates: &NodalReferenceCoordinates<3>,
    side_set: &SideSet,
    traction: &Traction,
) -> Result<NodalForcesSolid<3>, String> {
    if side_set.faces().len() != side_set.sides().len() {
        return Err("The side set has no faces on the mesh.".to_string());
    }
    let mut forces = NodalForcesSolid::zero(coordinates.len());
    side_set.faces().iter().try_for_each(|face| {
        let nodal_forces = match *face.as_slice() {
            [a, b, c] => {
                let reference = ElementNodalReferenceCoordinates::from([
                    coordinates[a].clone(),
                    coordinates[b].clone(),
                    coordinates[c].clone(),
                ]);
                Triangle::traction_forces(&reference, traction)
                    .into_iter()
                    .collect::<Vec<Force>>()
            }
            [a, b, c, d] => {
                let reference = ElementNodalReferenceCoordinates::from([
                    coordinates[a].clone(),
                    coordinates[b].clone(),
                    coordinates[c].clone(),
                    coordinates[d].clone(),
                ]);
                Quadrilateral::traction_forces(&reference, traction)
                    .into_iter()
                    .collect::<Vec<Force>>()
            }
            _ => {
                return Err(format!(
                    "A face with {} nodes is not a triangle or a quadrilateral.",
                    face.len()
                ));
            }
        };
        face.iter()
            .zip(nodal_forces)
            .for_each(|(&node, force)| forces[node] += force);
        Ok(())
    })?;
    Ok(forces)
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
    coordinates: NodalReferenceCoordinates<3>,
    side_set: SideSet,
    traction: G,
}

type WithTimedTraction<T, G> = Loads<Solid<3>, (T, TimedTraction<G>)>;

impl<T> Loads<Solid<3>, T> {
    /// Adds a dead traction per unit reference area on the faces of a side set, that is a
    /// function of time.
    pub fn traction_with(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        traction: impl Fn(Quantity<Time>) -> Traction,
    ) -> Result<WithTimedTraction<T, impl Fn(Quantity<Time>) -> Traction>, String> {
        traction_forces(coordinates, side_set, &Traction::from([0.0; 3]))?;
        Ok(self.with(TimedTraction {
            coordinates: coordinates.clone(),
            side_set: side_set.clone(),
            traction,
        }))
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
        if nodal_coordinates.len() != self.coordinates.len() {
            return Err("The traction is on a mesh with another number of nodes.".to_string());
        }
        let forces = traction_forces(&self.coordinates, &self.side_set, &(self.traction)(time))?;
        residual
            .iter_mut()
            .zip(forces)
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
        if nodal_coordinates.len() != self.coordinates.len() {
            return Err("The traction is on a mesh with another number of nodes.".to_string());
        }
        let forces = traction_forces(&self.coordinates, &self.side_set, &(self.traction)(time))?;
        let reference: NodalCoordinates<3> = self.coordinates.clone().into();
        Ok(-nodal_coordinates
            .iter()
            .zip(reference)
            .zip(forces)
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
        if nodal_coordinates.len() != self.coordinates.len() {
            return Err("The traction is on a mesh with another number of nodes.".to_string());
        }
        let forces = traction_forces(&self.coordinates, &self.side_set, &(self.traction)(time))?;
        Ok(-nodal_velocities
            .iter()
            .zip(forces)
            .map(|(velocity, force)| force * velocity)
            .sum::<Quantity<Power>>())
    }
}
