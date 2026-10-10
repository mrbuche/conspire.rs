use crate::{
    cbm::{
        block::{Block, node::thermal::conduction::ThermalConductionNode},
        thermal::conduction::ThermalConductionElements,
    },
    constitutive::{ConstitutiveError, thermal::conduction::ThermalConduction},
    domain::ElementModelError,
    math::Quantity,
    units::PowerTemperature,
};

pub use crate::domain::thermal::{NodalForcesThermal, NodalStiffnessesThermal, NodalTemperatures};

impl<C, R> ThermalConductionElements for Block<C, R>
where
    C: ThermalConduction,
{
    fn potential(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Quantity<PowerTemperature>, ElementModelError> {
        self.nodes
            .iter()
            .map(|node| node.potential(&self.constitutive_model, nodal_temperatures))
            .sum::<Result<Quantity<PowerTemperature>, ConstitutiveError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_forces_into(
        &self,
        nodal_temperatures: &NodalTemperatures,
        nodal_forces: &mut NodalForcesThermal,
    ) -> Result<(), ElementModelError> {
        self.nodes
            .iter()
            .try_for_each(|node| {
                node.nodal_forces(&self.constitutive_model, nodal_temperatures)?
                    .into_iter()
                    .zip(node.neighbors())
                    .for_each(|(force, &neighbor)| nodal_forces[neighbor] += force);
                Ok::<(), ConstitutiveError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_temperatures: &NodalTemperatures,
        nodal_stiffnesses: &mut NodalStiffnessesThermal,
    ) -> Result<(), ElementModelError> {
        self.nodes
            .iter()
            .try_for_each(|node| {
                node.nodal_stiffnesses(&self.constitutive_model, nodal_temperatures)?
                    .into_iter()
                    .zip(node.neighbors())
                    .for_each(|(row, &neighbor_a)| {
                        row.into_iter().zip(node.neighbors()).for_each(
                            |(stiffness, &neighbor_b)| {
                                nodal_stiffnesses[neighbor_a][neighbor_b] += stiffness
                            },
                        )
                    });
                Ok::<(), ConstitutiveError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
