#[cfg(test)]
mod test;

use crate::{
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
    math::Tensor,
    mechanics::{Force, Traction},
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
