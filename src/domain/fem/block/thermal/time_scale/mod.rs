use crate::{
    constitutive::thermal::conduction::ThermalConduction,
    domain::{
        thermal::time_scale::ThermalTimeScaleElements,
        time_scale::{diffusive_time_scale_from_eigenvalue, largest_eigenvalue},
    },
    fem::{
        ElementModelError, NodalReferenceCoordinates,
        block::{
            Block,
            element::{
                FiniteElementError,
                thermal::{
                    capacity::LumpedHeatCapacityFiniteElement,
                    conduction::ThermalConductionFiniteElement,
                },
            },
            thermal::{NodalTemperatures, ThermalElements, capacity::HeatCapacities},
        },
    },
    math::{Quantity, Tensor},
    units::Time,
};

impl<C, F, R, H, const G: usize, const M: usize, const N: usize, const P: usize>
    ThermalTimeScaleElements<M> for Block<C, F, G, M, N, P, R, H>
where
    C: ThermalConduction,
    F: ThermalConductionFiniteElement<C, G, M, N, P> + LumpedHeatCapacityFiniteElement<G, M, N, P>,
    H: HeatCapacities<G>,
{
    fn fastest_diffusive_time_scale(
        &self,
        _reference_coordinates: &NodalReferenceCoordinates<M>,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Quantity<Time>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .enumerate()
            .try_fold(
                Time::seconds(f64::INFINITY),
                |fastest, (element_index, (element, element_connectivity))| {
                    let stiffnesses = element.nodal_stiffnesses(
                        self.constitutive_model(),
                        &self.nodal_temperatures_element(element_connectivity, nodal_temperatures),
                    )?;
                    let capacities: Vec<f64> = element
                        .nodal_lumped_heat_capacities(&self.heat_capacity().at(element_index))
                        .iter()
                        .map(|capacity| capacity.value())
                        .collect();
                    let eigenvalue = largest_eigenvalue(
                        N,
                        |row, column| stiffnesses[row][column].value(),
                        &capacities,
                    );
                    Ok::<_, FiniteElementError>(
                        fastest.min(diffusive_time_scale_from_eigenvalue(eigenvalue)),
                    )
                },
            )
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
