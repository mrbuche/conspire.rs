use crate::{
    cbm::block::{Block, node::thermal::capacity::NodalHeatCapacity},
    domain::thermal::capacity::{LumpedHeatCapacityElements, NodalLumpedHeatCapacities},
    math::Quantity,
    units::VolumetricHeatCapacity,
};

pub type NodalHeatCapacities = Vec<Quantity<VolumetricHeatCapacity>>;

pub trait HeatCapacities {
    fn at(&self, node: usize) -> Quantity<VolumetricHeatCapacity>;
}

impl HeatCapacities for NodalHeatCapacities {
    fn at(&self, node: usize) -> Quantity<VolumetricHeatCapacity> {
        self[node]
    }
}

impl HeatCapacities for Quantity<VolumetricHeatCapacity> {
    fn at(&self, _node: usize) -> Quantity<VolumetricHeatCapacity> {
        *self
    }
}

impl<C, R> Block<C, R> {
    pub fn with_heat_capacity<S>(self, heat_capacity: S) -> Block<C, S>
    where
        S: HeatCapacities,
    {
        self.with_density(heat_capacity)
    }
}

impl<C, R> LumpedHeatCapacityElements for Block<C, R>
where
    R: HeatCapacities,
{
    fn nodal_lumped_heat_capacities_into(
        &self,
        nodal_lumped_heat_capacities: &mut NodalLumpedHeatCapacities,
    ) {
        self.nodes.iter().enumerate().for_each(|(index, node)| {
            nodal_lumped_heat_capacities[index] += node.nodal_heat_capacity(self.density.at(index))
        })
    }
}
