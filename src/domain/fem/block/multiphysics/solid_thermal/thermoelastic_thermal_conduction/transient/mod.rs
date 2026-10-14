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
            NodalForcesSolidThermal, NodalHeatingsSolidThermal, NodalStiffnessesSolidThermal,
            thermoelastic_thermal_conduction::transient::ThermoelasticThermalConductionTransientElements,
        },
    },
    math::TensorTuple,
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
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
        nodal_temperatures: &NodalTemperatures,
        nodal_stiffnesses: &mut NodalStiffnessesSolidThermal,
        nodal_heatings: &mut NodalHeatingsSolidThermal,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, nodes)| {
                let ((solid, coupling, thermal), heating_stiffnesses, heatings) = element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                        &Self::element_coordinates(nodal_velocities, nodes),
                        &self.nodal_temperatures_element(nodes, nodal_temperatures),
                    )?;
                solid
                    .into_iter()
                    .zip(coupling)
                    .zip(thermal)
                    .zip(heating_stiffnesses)
                    .zip(heatings)
                    .zip(nodes)
                    .for_each(
                        |(
                            ((((solid, coupling), thermal), heating_stiffnesses), heatings),
                            &node_a,
                        )| {
                            solid
                                .into_iter()
                                .zip(coupling)
                                .zip(thermal)
                                .zip(heating_stiffnesses)
                                .zip(heatings)
                                .zip(nodes)
                                .for_each(
                                    |(
                                        (
                                            (((solid, coupling), thermal), heating_stiffness),
                                            heating,
                                        ),
                                        &node_b,
                                    )| {
                                        nodal_stiffnesses[node_a][node_b] += TensorTuple(
                                            TensorTuple(solid, coupling),
                                            TensorTuple(heating_stiffness, thermal),
                                        );
                                        nodal_heatings[node_a][node_b] += heating
                                    },
                                )
                        },
                    );
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
