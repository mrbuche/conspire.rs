use super::{NodalTemperatures, capacity::HeatCapacities};
use crate::{
    cbm::block::{
        Block,
        node::{diffusion::DiffusionNode, thermal::capacity::NodalHeatCapacity},
    },
    constitutive::{ConstitutiveError, thermal::conduction::ThermalConduction},
    domain::{
        ElementModelError, NodalReferenceCoordinates,
        thermal::time_scale::ThermalTimeScaleElements,
        time_scale::{diffusive_time_scale_from_eigenvalue, largest_eigenvalue},
    },
    math::{Quantity, Scalar},
    units::Time,
};

impl<C, R> ThermalTimeScaleElements<3> for Block<C, R>
where
    C: ThermalConduction,
    R: HeatCapacities,
{
    fn fastest_diffusive_time_scale(
        &self,
        _reference_coordinates: &NodalReferenceCoordinates<3>,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Quantity<Time>, ElementModelError> {
        self.nodes
            .iter()
            .try_fold(0.0_f64, |largest, node| {
                let stiffnesses =
                    node.nodal_stiffnesses(&self.constitutive_model, nodal_temperatures)?;
                let capacities: Vec<Scalar> = node
                    .neighbors()
                    .iter()
                    .map(|&neighbor| {
                        self.nodes[neighbor]
                            .nodal_heat_capacity(self.density.at(neighbor))
                            .value()
                            / self.nodes[neighbor].neighbors().len() as Scalar
                    })
                    .collect();
                Ok::<_, ConstitutiveError>(largest.max(largest_eigenvalue(
                    capacities.len(),
                    |row, column| stiffnesses[row][column].value(),
                    &capacities,
                )))
            })
            .map(diffusive_time_scale_from_eigenvalue)
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
