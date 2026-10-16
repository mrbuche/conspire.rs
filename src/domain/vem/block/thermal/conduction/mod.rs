#[cfg(test)]
mod test;

use crate::{
    constitutive::thermal::conduction::ThermalConduction,
    domain::{ElementModelError, diffusion::DiffusionElements},
    math::{Quantity, Tensor},
    units::{PowerTemperature, Temperature},
    vem::block::{
        Block,
        element::{VirtualElementError, thermal::conduction::ThermalConductionVirtualElement},
        thermal::NodalTemperatures,
    },
};

pub use crate::domain::thermal::{NodalForcesThermal, NodalStiffnessesThermal};

impl<C, F, R> DiffusionElements<Temperature> for Block<C, F, R>
where
    C: ThermalConduction,
    F: ThermalConductionVirtualElement<C>,
{
    fn potential(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Quantity<PowerTemperature>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .map(|(element, nodes)| {
                element.potential(
                    self.constitutive_model(),
                    &Self::element_temperatures(nodes, nodal_temperatures),
                )
            })
            .sum::<Result<Quantity<PowerTemperature>, VirtualElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_forces_into(
        &self,
        nodal_temperatures: &NodalTemperatures,
        nodal_forces: &mut NodalForcesThermal,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &Self::element_temperatures(nodes, nodal_temperatures),
                    )?
                    .iter()
                    .zip(nodes)
                    .for_each(|(&nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), VirtualElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_temperatures: &NodalTemperatures,
        nodal_stiffnesses: &mut NodalStiffnessesThermal,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_temperatures(nodes, nodal_temperatures),
                    )?
                    .iter()
                    .zip(nodes)
                    .for_each(|(row, &node_a)| {
                        row.iter()
                            .zip(nodes)
                            .for_each(|(&nodal_stiffness, &node_b)| {
                                nodal_stiffnesses[node_a][node_b] += nodal_stiffness
                            })
                    });
                Ok::<(), VirtualElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
