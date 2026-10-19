#[cfg(test)]
mod test;

use crate::{
    domain::{
        boundary::Loads,
        solid::loads::{DeadLoad, SolidLayout, TimedLoad},
    },
    fem::{
        NodalReferenceCoordinates,
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
    units::{Area, Time},
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
pub type TimedTraction<G> = TimedLoad<Area, G>;

type WithTimedTraction<L, T, G> = Result<Loads<L, (T, TimedTraction<G>)>, String>;

impl<L, T> Loads<L, T>
where
    L: SolidLayout,
{
    /// Adds a dead traction per unit reference area on the faces of a side set, that is a
    /// function of time.
    pub fn traction_with<G>(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        traction: G,
    ) -> WithTimedTraction<L, T, G>
    where
        G: Fn(Quantity<Time>) -> Traction,
    {
        Ok(self.with(TimedLoad::new(
            coordinates,
            nodal_areas(coordinates, side_set)?,
            traction,
        )?))
    }
}
