use crate::{
    constitutive::thermal::conduction::ThermalConduction,
    domain::{
        ElementModelError,
        thermal::time_scale::ThermalTimeScaleElements,
        time_scale::{diffusive_time_scale_from_eigenvalue, largest_eigenvalue},
    },
    math::{Quantity, Tensor},
    units::Time,
    vem::{
        NodalReferenceCoordinates,
        block::{
            Block,
            element::{
                VirtualElementError,
                thermal::{
                    capacity::LumpedHeatCapacityVirtualElement,
                    conduction::ThermalConductionVirtualElement,
                },
            },
            thermal::{NodalTemperatures, capacity::HeatCapacities},
        },
    },
};

impl<C, F, R> ThermalTimeScaleElements<3> for Block<C, F, R>
where
    C: ThermalConduction,
    F: ThermalConductionVirtualElement<C> + LumpedHeatCapacityVirtualElement,
    R: HeatCapacities,
{
    fn fastest_diffusive_time_scale(
        &self,
        reference_coordinates: &NodalReferenceCoordinates,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Quantity<Time>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .enumerate()
            .try_fold(
                Time::seconds(f64::INFINITY),
                |fastest, (element_index, (element, nodes))| {
                    let stiffnesses = element.nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_temperatures(nodes, nodal_temperatures),
                    )?;
                    let capacities: Vec<f64> = element
                        .nodal_lumped_heat_capacities(
                            self.density().at(element_index),
                            &Self::element_coordinates(reference_coordinates, nodes),
                        )
                        .iter()
                        .map(|capacity| capacity.value())
                        .collect();
                    let eigenvalue = largest_eigenvalue(
                        nodes.len(),
                        |row, column| stiffnesses[row][column].value(),
                        &capacities,
                    );
                    Ok::<_, VirtualElementError>(
                        fastest.min(diffusive_time_scale_from_eigenvalue(eigenvalue)),
                    )
                },
            )
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
