use crate::{
    domain::thermal::capacity::{LumpedHeatCapacityElements, NodalLumpedHeatCapacities},
    math::{Quantity, Tensor},
    units::VolumetricHeatCapacity,
    vem::{
        NodalReferenceCoordinates,
        block::{Block, element::thermal::capacity::LumpedHeatCapacityVirtualElement},
    },
};

pub type ElementHeatCapacities = Vec<Quantity<VolumetricHeatCapacity>>;

pub trait HeatCapacities {
    fn at(&self, element: usize) -> Quantity<VolumetricHeatCapacity>;
}

impl HeatCapacities for ElementHeatCapacities {
    fn at(&self, element: usize) -> Quantity<VolumetricHeatCapacity> {
        self[element]
    }
}

impl HeatCapacities for Quantity<VolumetricHeatCapacity> {
    fn at(&self, _element: usize) -> Quantity<VolumetricHeatCapacity> {
        *self
    }
}

impl<C, F, R> Block<C, F, R> {
    pub fn with_heat_capacity<S>(self, heat_capacity: S) -> Block<C, F, S>
    where
        S: HeatCapacities,
    {
        self.with_density(heat_capacity)
    }
}

impl<C, F, R> LumpedHeatCapacityElements<3> for Block<C, F, R>
where
    F: LumpedHeatCapacityVirtualElement,
    R: HeatCapacities,
{
    fn nodal_lumped_heat_capacities_into(
        &self,
        reference_coordinates: &NodalReferenceCoordinates,
        nodal_lumped_heat_capacities: &mut NodalLumpedHeatCapacities,
    ) {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .enumerate()
            .for_each(|(element_index, (element, nodes))| {
                element
                    .nodal_lumped_heat_capacities(
                        self.density().at(element_index),
                        &Self::element_coordinates(reference_coordinates, nodes),
                    )
                    .iter()
                    .zip(nodes)
                    .for_each(|(&nodal_heat_capacity, &node)| {
                        nodal_lumped_heat_capacities[node] += nodal_heat_capacity
                    })
            })
    }
}
