#[cfg(test)]
mod test;

use crate::{
    domain::mass::factor_free,
    fem::{Blocks, ElementModel, Elements, Model},
    math::{
        Quantity, QuantitySparseVec2D, QuantityVector, Tensor, Vector,
        sparse::{CscLdl, SparseError},
    },
    units::{HeatCapacity, Power},
};

pub type NodalHeatCapacities = QuantitySparseVec2D<HeatCapacity>;
pub type NodalLumpedHeatCapacities = QuantityVector<HeatCapacity>;

pub trait ConsistentHeatCapacityElements
where
    Self: Elements,
{
    fn nodal_heat_capacities_into(&self, nodal_heat_capacities: &mut NodalHeatCapacities);
}

pub trait LumpedHeatCapacityElements
where
    Self: Elements,
{
    fn nodal_lumped_heat_capacities_into(
        &self,
        nodal_lumped_heat_capacities: &mut NodalLumpedHeatCapacities,
    );
}

impl<B1, B2> ConsistentHeatCapacityElements for Blocks<B1, B2>
where
    B1: ConsistentHeatCapacityElements,
    B2: ConsistentHeatCapacityElements,
{
    fn nodal_heat_capacities_into(&self, nodal_heat_capacities: &mut NodalHeatCapacities) {
        self.0.nodal_heat_capacities_into(nodal_heat_capacities);
        self.1.nodal_heat_capacities_into(nodal_heat_capacities)
    }
}

impl<B1, B2> LumpedHeatCapacityElements for Blocks<B1, B2>
where
    B1: LumpedHeatCapacityElements,
    B2: LumpedHeatCapacityElements,
{
    fn nodal_lumped_heat_capacities_into(
        &self,
        nodal_lumped_heat_capacities: &mut NodalLumpedHeatCapacities,
    ) {
        self.0
            .nodal_lumped_heat_capacities_into(nodal_lumped_heat_capacities);
        self.1
            .nodal_lumped_heat_capacities_into(nodal_lumped_heat_capacities)
    }
}

impl<B, const D: usize> Model<B, D>
where
    B: ConsistentHeatCapacityElements,
{
    pub fn nodal_heat_capacities(&self) -> NodalHeatCapacities {
        let mut nodal_heat_capacities = NodalHeatCapacities::zero(self.coordinates().len());
        self.blocks
            .nodal_heat_capacities_into(&mut nodal_heat_capacities);
        nodal_heat_capacities
    }
}

impl<B, const D: usize> Model<B, D>
where
    B: LumpedHeatCapacityElements,
{
    pub fn nodal_lumped_heat_capacities(&self) -> NodalLumpedHeatCapacities {
        let mut nodal_lumped_heat_capacities =
            NodalLumpedHeatCapacities::zero(self.coordinates().len());
        self.blocks
            .nodal_lumped_heat_capacities_into(&mut nodal_lumped_heat_capacities);
        nodal_lumped_heat_capacities
    }
}

/// The factors of a consistent heat capacity restricted to the free temperatures.
pub struct FreeFactoredHeatCapacities {
    factors: CscLdl,
    free: Vec<usize>,
}

impl NodalHeatCapacities {
    /// Factors the heat capacity restricted to the nodes that are not fixed.
    pub fn factor(&self, fixed: &[usize]) -> Result<FreeFactoredHeatCapacities, SparseError> {
        let (factors, free) = factor_free::<HeatCapacity, 1>(self, fixed)?;
        Ok(FreeFactoredHeatCapacities { factors, free })
    }
}

impl FreeFactoredHeatCapacities {
    /// The temperature rates given the heat at every node, which vanish at fixed nodes.
    pub fn nodal_temperature_rates(&self, heat: &QuantityVector<Power>) -> Vector {
        let heating: Vector = self.free.iter().map(|&index| heat[index].value()).collect();
        let free_rates = self.factors.solve(&heating);
        let mut rates = Vector::zero(heat.len());
        self.free
            .iter()
            .enumerate()
            .for_each(|(k, &index)| rates[index] = free_rates[k]);
        rates
    }
}

impl NodalLumpedHeatCapacities {
    pub fn total(&self) -> Quantity<HeatCapacity> {
        self.iter().copied().sum()
    }
}
