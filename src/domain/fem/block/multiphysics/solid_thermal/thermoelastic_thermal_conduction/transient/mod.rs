#[cfg(test)]
pub mod test;

use crate::{
    constitutive::multiphysics::ThermoelasticThermalConduction,
    fem::{
        ElementModelError, NodalCoordinates, NodalVelocities,
        block::{
            Block,
            element::{
                FiniteElementError,
                multiphysics::solid_thermal::thermoelastic_thermal_conduction::transient::ThermoelasticThermalConductionTransientFiniteElement,
            },
            thermal::{NodalTemperatures, ThermalElements},
        },
        multiphysics::solid_thermal::{
            NodalForcesSolidThermal,
            thermoelastic_thermal_conduction::transient::ThermoelasticThermalConductionTransientElements,
        },
    },
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, R, H>
    ThermoelasticThermalConductionTransientElements for Block<C, F, G, M, N, P, R, H>
where
    C: ThermoelasticThermalConduction,
    F: ThermoelasticThermalConductionTransientFiniteElement<C, G, M, N, P>,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
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
                        &Self::element_coordinates(nodal_velocities, nodes),
                        &self.nodal_temperatures_element(nodes, nodal_temperatures),
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
