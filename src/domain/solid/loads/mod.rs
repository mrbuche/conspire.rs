use crate::{
    domain::{
        NodalCoordinates, NodalReferenceCoordinates,
        boundary::{Conservative, Layout, Load, LoadAt, Loads, Solid, SolidThermal},
        multiphysics::solid_thermal::{NodalForcesSolidThermal, NodalStatesSolidThermal},
        solid::NodalForcesSolid,
    },
    math::{Quantity, Tensor},
    units::{Energy, Time},
};

/// A layout whose degrees of freedom begin with the coordinates of a solid in three dimensions.
pub trait SolidLayout: Layout {}

impl SolidLayout for Solid<3> {}

impl SolidLayout for SolidThermal<3> {}

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
    fn check(&self, nodes: usize) -> Result<(), String> {
        if nodes == self.reference.len() {
            Ok(())
        } else {
            Err("The load is on a mesh with another number of nodes.".to_string())
        }
    }
}

impl<L, T> Loads<L, T>
where
    L: SolidLayout,
{
    /// Adds a force on each node, such as the weight of a body.
    pub fn nodal_forces(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        forces: NodalForcesSolid<3>,
    ) -> Result<Loads<L, (T, DeadLoad)>, String> {
        Ok(self.with(DeadLoad::new(coordinates, forces)?))
    }
}

impl Load<NodalCoordinates<3>, NodalForcesSolid<3>> for DeadLoad {
    fn subtract_from(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        residual: &mut NodalForcesSolid<3>,
    ) -> Result<(), String> {
        self.check(nodal_coordinates.len())?;
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
        self.check(nodal_coordinates.len())?;
        Ok(-nodal_coordinates
            .iter()
            .zip(&self.reference)
            .zip(&self.forces)
            .map(|((coordinate, reference), force)| force * &(coordinate - reference))
            .sum::<Quantity<Energy>>())
    }
}

impl Load<NodalStatesSolidThermal, NodalForcesSolidThermal> for DeadLoad {
    fn subtract_from(
        &self,
        nodal_states: &NodalStatesSolidThermal,
        residual: &mut NodalForcesSolidThermal,
    ) -> Result<(), String> {
        self.check(nodal_states.len())?;
        residual
            .iter_mut()
            .zip(&self.forces)
            .for_each(|(residual, force)| residual.0 -= force);
        Ok(())
    }
}

impl LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal> for DeadLoad {
    fn subtract_at(
        &self,
        _: Quantity<Time>,
        nodal_states: &NodalStatesSolidThermal,
        residual: &mut NodalForcesSolidThermal,
    ) -> Result<(), String> {
        self.subtract_from(nodal_states, residual)
    }
}
