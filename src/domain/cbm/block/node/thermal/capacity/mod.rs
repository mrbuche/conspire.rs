use super::super::Node;
use crate::{
    math::Quantity,
    units::{HeatCapacity, VolumetricHeatCapacity},
};

pub trait NodalHeatCapacity {
    fn nodal_heat_capacity(
        &self,
        volumetric_heat_capacity: Quantity<VolumetricHeatCapacity>,
    ) -> Quantity<HeatCapacity>;
}

impl NodalHeatCapacity for Node {
    fn nodal_heat_capacity(
        &self,
        volumetric_heat_capacity: Quantity<VolumetricHeatCapacity>,
    ) -> Quantity<HeatCapacity> {
        volumetric_heat_capacity * self.volume
    }
}
