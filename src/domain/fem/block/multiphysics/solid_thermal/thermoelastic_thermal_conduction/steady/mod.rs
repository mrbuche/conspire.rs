#[cfg(test)]
pub mod test;

use crate::{
    constitutive::{
        multiphysics::ThermoelasticThermalConduction, solid::thermoelastic::Thermoelastic,
    },
    fem::{
        ElementModelError, NodalCoordinates,
        block::{
            Block,
            element::{
                FiniteElementError,
                multiphysics::solid_thermal::thermoelastic_thermal_conduction::steady::ThermoelasticThermalConductionFiniteElement,
            },
            thermal::{NodalTemperatures, ThermalElements},
        },
        multiphysics::solid_thermal::{
            NodalForcesSolidThermal, NodalStiffnessesSolidThermal,
            thermoelastic_thermal_conduction::steady::ThermoelasticThermalConductionElements,
        },
    },
    math::TensorTuple,
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, R, H>
    ThermoelasticThermalConductionElements for Block<C, F, G, M, N, P, R, H>
where
    C: ThermoelasticThermalConduction,
    F: ThermoelasticThermalConductionFiniteElement<C, G, M, N, P>,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_forces: &mut NodalForcesSolidThermal,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                        &self.nodal_temperatures_element(nodes, nodal_temperatures),
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_reference_temperatures_into(&self, nodal_temperatures: &mut NodalTemperatures) {
        let reference_temperature = self
            .constitutive_model()
            .solid_constitutive_model()
            .reference_temperature();
        self.connectivity().iter().for_each(|nodes| {
            nodes
                .iter()
                .for_each(|&node| nodal_temperatures[node] = reference_temperature)
        })
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_stiffnesses: &mut NodalStiffnessesSolidThermal,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, nodes)| {
                let (solid, coupling, thermal) = element.nodal_stiffnesses(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    &self.nodal_temperatures_element(nodes, nodal_temperatures),
                )?;
                solid
                    .into_iter()
                    .zip(coupling)
                    .zip(thermal)
                    .zip(nodes)
                    .for_each(|(((solid, coupling), thermal), &node_a)| {
                        solid
                            .into_iter()
                            .zip(coupling)
                            .zip(thermal)
                            .zip(nodes)
                            .for_each(|(((solid, coupling), thermal), &node_b)| {
                                nodal_stiffnesses[node_a][node_b] += TensorTuple(
                                    TensorTuple(solid, coupling),
                                    TensorTuple(Default::default(), thermal),
                                )
                            })
                    });
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
