use crate::{
    domain::{
        NodalCoordinates, NodalReferenceCoordinates,
        boundary::{Conservative, Load, LoadAt, Loads, Solid},
        solid::NodalForcesSolid,
    },
    math::{Quantity, Tensor},
    units::{Energy, Time},
};

/// Forces on the nodes of a solid that do not change as the body deforms.
#[derive(Clone, Debug)]
pub struct DeadLoad {
    reference: NodalCoordinates<3>,
    forces: NodalForcesSolid<3>,
}

impl DeadLoad {
    pub(crate) fn new(
        coordinates: &NodalReferenceCoordinates<3>,
        forces: NodalForcesSolid<3>,
    ) -> Result<Self, String> {
        if coordinates.len() == forces.len() {
            Ok(Self {
                reference: coordinates.clone().into(),
                forces,
            })
        } else {
            Err("There must be exactly one force for each node.".to_string())
        }
    }
    fn check(&self, nodal_coordinates: &NodalCoordinates<3>) -> Result<(), String> {
        if nodal_coordinates.len() == self.reference.len() {
            Ok(())
        } else {
            Err("The load is on a mesh with another number of nodes.".to_string())
        }
    }
}

impl<T> Loads<Solid<3>, T> {
    /// Adds a force on each node, such as the weight of a body.
    pub fn nodal_forces(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        forces: NodalForcesSolid<3>,
    ) -> Result<Loads<Solid<3>, (T, DeadLoad)>, String> {
        Ok(self.with(DeadLoad::new(coordinates, forces)?))
    }
}

impl Load<NodalCoordinates<3>, NodalForcesSolid<3>> for DeadLoad {
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

impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>> for DeadLoad {
    fn subtract_at(
        &self,
        _: Quantity<Time>,
        nodal_coordinates: &NodalCoordinates<3>,
        residual: &mut NodalForcesSolid<3>,
    ) -> Result<(), String> {
        self.subtract_from(nodal_coordinates, residual)
    }
}

impl Conservative<NodalCoordinates<3>, Quantity<Energy>> for DeadLoad {
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
