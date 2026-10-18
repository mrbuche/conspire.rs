#[cfg(test)]
mod test;

use crate::{
    domain::boundary::{Conservative, Load, Loads, Solid},
    fem::{
        NodalCoordinates, NodalReferenceCoordinates,
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
    units::Energy,
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

/// A traction per unit reference area on the faces of a side set, whose force does not change
/// as the body deforms.
#[derive(Clone, Debug)]
pub struct DeadTraction {
    reference: NodalCoordinates<3>,
    forces: NodalForcesSolid<3>,
}

impl<T> Loads<Solid<3>, T> {
    /// Adds a dead traction per unit reference area on the faces of a side set.
    pub fn traction(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        traction: &Traction,
    ) -> Result<Loads<Solid<3>, (T, DeadTraction)>, String> {
        Ok(self.with(DeadTraction {
            reference: coordinates.clone().into(),
            forces: traction_forces(coordinates, side_set, traction)?,
        }))
    }
}

impl DeadTraction {
    fn check(&self, nodal_coordinates: &NodalCoordinates<3>) -> Result<(), String> {
        if nodal_coordinates.len() == self.reference.len() {
            Ok(())
        } else {
            Err("The traction is on a mesh with another number of nodes.".to_string())
        }
    }
}

impl Load<NodalCoordinates<3>, NodalForcesSolid<3>> for DeadTraction {
    fn subtract_from(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        residual: &mut NodalForcesSolid<3>,
    ) -> Result<(), String> {
        self.check(nodal_coordinates)?;
        residual
            .iter_mut()
            .zip(&self.forces)
            .for_each(|(residual, force)| *residual -= force);
        Ok(())
    }
}

impl Conservative<NodalCoordinates<3>, Quantity<Energy>> for DeadTraction {
    fn potential_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Energy>, String> {
        self.check(nodal_coordinates)?;
        Ok(-nodal_coordinates
            .iter()
            .zip(&self.reference)
            .zip(&self.forces)
            .map(|((coordinate, reference), force)| force * &(coordinate - reference))
            .sum::<Quantity<Energy>>())
    }
}
