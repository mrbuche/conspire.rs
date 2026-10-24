#[cfg(test)]
pub mod test;

use crate::{
    constitutive::multiphysics::ThermoelasticThermalConduction,
    domain::time_scale::{diffusive_time_scale_from_eigenvalue, largest_eigenvalue},
    fem::{
        ElementModelError, NodalCoordinates, NodalReferenceCoordinates,
        block::{
            Block, Densities, HeatCapacities,
            element::{
                FiniteElementError, mass::LumpedMassFiniteElement,
                multiphysics::solid_thermal::thermoelastic_thermal_conduction::steady::ThermoelasticThermalConductionFiniteElement,
                thermal::capacity::LumpedHeatCapacityFiniteElement, time_scale::fastest_time_scale,
            },
            thermal::{NodalTemperatures, ThermalElements},
        },
        multiphysics::solid_thermal::thermoelastic_thermal_conduction::dynamics::time_scale::ThermoelasticThermalConductionTimeScaleElements,
    },
    math::{Quantity, Tensor},
    units::Time,
};

impl<C, F, R, H, const G: usize, const M: usize, const N: usize>
    ThermoelasticThermalConductionTimeScaleElements for Block<C, F, G, M, N, R, H>
where
    C: ThermoelasticThermalConduction,
    F: ThermoelasticThermalConductionFiniteElement<C, G, M, N>
        + LumpedMassFiniteElement<G, M, N>
        + LumpedHeatCapacityFiniteElement<G, M, N>,
    R: Densities<G>,
    H: HeatCapacities<G>,
{
    fn fastest_time_scales(
        &self,
        _reference_coordinates: &NodalReferenceCoordinates<3>,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<(Quantity<Time>, Quantity<Time>), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .enumerate()
            .try_fold(
                (Time::seconds(f64::INFINITY), Time::seconds(f64::INFINITY)),
                |(wave, diffusion), (element_index, (element, nodes))| {
                    let (solid, _, thermal) = element.nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                        &self.nodal_temperatures_element(nodes, nodal_temperatures),
                    )?;
                    let masses = element.nodal_lumped_masses(&self.density().at(element_index));
                    let capacities: Vec<f64> = element
                        .nodal_lumped_heat_capacities(&self.heat_capacity().at(element_index))
                        .iter()
                        .map(|capacity| capacity.value())
                        .collect();
                    let eigenvalue = largest_eigenvalue(
                        N,
                        |row, column| thermal[row][column].value(),
                        &capacities,
                    );
                    Ok::<_, FiniteElementError>((
                        wave.min(fastest_time_scale(&solid, &masses)),
                        diffusion.min(diffusive_time_scale_from_eigenvalue(eigenvalue)),
                    ))
                },
            )
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
