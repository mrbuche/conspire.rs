#[cfg(test)]
mod test;

use crate::{
    math::{Quantity, TensorVector},
    units::{HeatCapacity, VolumetricHeatCapacity},
    vem::{
        NodalReferenceCoordinates,
        block::element::{Element, VirtualElement, mass::lumping_fractions},
    },
};

pub type ElementNodalLumpedHeatCapacities = TensorVector<Quantity<HeatCapacity>>;

pub trait LumpedHeatCapacityVirtualElement
where
    Self: VirtualElement,
{
    fn nodal_lumped_heat_capacities(
        &self,
        capacity: Quantity<VolumetricHeatCapacity>,
        nodal_reference_coordinates: &NodalReferenceCoordinates,
    ) -> ElementNodalLumpedHeatCapacities;
}

impl LumpedHeatCapacityVirtualElement for Element {
    fn nodal_lumped_heat_capacities(
        &self,
        capacity: Quantity<VolumetricHeatCapacity>,
        nodal_reference_coordinates: &NodalReferenceCoordinates,
    ) -> ElementNodalLumpedHeatCapacities {
        let total = capacity * self.volume();
        lumping_fractions(self, nodal_reference_coordinates)
            .into_iter()
            .map(|fraction| total * fraction)
            .collect()
    }
}
