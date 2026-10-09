#[cfg(test)]
mod test;

use crate::{
    domain::mass::factor_free,
    fem::block::thermal::conduction::NodalForcesThermal,
    fem::{Blocks, ElementModel, Elements, Model},
    math::{
        Erase, Quantity, QuantitySparseVec2D, QuantityVector, Tensor, Vector,
        sparse::{CscLdl, SparseError},
    },
    units::{HeatCapacity, TemperatureRate},
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

pub type NodalTemperatureRates = QuantityVector<TemperatureRate>;

/// A heat capacity inverted for the rates of temperature it gives.
pub trait InverseHeatCapacity {
    fn nodal_temperature_rates(
        &self,
        external_heating: &NodalForcesThermal,
        internal_heating: &NodalForcesThermal,
    ) -> NodalTemperatureRates;
}

impl InverseHeatCapacity for NodalLumpedHeatCapacities {
    fn nodal_temperature_rates(
        &self,
        external_heating: &NodalForcesThermal,
        internal_heating: &NodalForcesThermal,
    ) -> NodalTemperatureRates {
        self.iter()
            .zip(external_heating.iter().zip(internal_heating.iter()))
            .map(|(&capacity, (&external, &internal))| (external - internal) / capacity)
            .collect()
    }
}

/// A heat capacity that yields its inverse with some temperatures held fixed.
///
/// The rates of the inverse vanish at the fixed nodes, and elsewhere they account for the
/// fixed ones not changing.
pub trait HeatCapacityMatrix {
    type Inverse: InverseHeatCapacity;
    fn inverse(&self, fixed: &[usize]) -> Result<Self::Inverse, SparseError>;
}

/// The inverse of lumped heat capacities with fixed temperatures.
pub struct FixedLumpedHeatCapacities {
    capacities: NodalLumpedHeatCapacities,
    fixed: Vec<usize>,
}

impl HeatCapacityMatrix for NodalLumpedHeatCapacities {
    type Inverse = FixedLumpedHeatCapacities;
    fn inverse(&self, fixed: &[usize]) -> Result<Self::Inverse, SparseError> {
        Ok(FixedLumpedHeatCapacities {
            capacities: self.clone(),
            fixed: fixed.to_vec(),
        })
    }
}

impl InverseHeatCapacity for FixedLumpedHeatCapacities {
    fn nodal_temperature_rates(
        &self,
        external_heating: &NodalForcesThermal,
        internal_heating: &NodalForcesThermal,
    ) -> NodalTemperatureRates {
        let mut rates = self
            .capacities
            .nodal_temperature_rates(external_heating, internal_heating);
        self.fixed
            .iter()
            .for_each(|&index| rates[index] = Default::default());
        rates
    }
}

impl HeatCapacityMatrix for NodalHeatCapacities {
    type Inverse = FreeFactoredHeatCapacities;
    fn inverse(&self, fixed: &[usize]) -> Result<Self::Inverse, SparseError> {
        self.factor(fixed)
    }
}

impl InverseHeatCapacity for FreeFactoredHeatCapacities {
    fn nodal_temperature_rates(
        &self,
        external_heating: &NodalForcesThermal,
        internal_heating: &NodalForcesThermal,
    ) -> NodalTemperatureRates {
        let (external, internal) = (external_heating.erase(), internal_heating.erase());
        let free_heating: Vector = self
            .free
            .iter()
            .map(|&index| external[index] - internal[index])
            .collect();
        let free_rates = self.factors.solve(&free_heating);
        let mut rates = Vector::zero(external_heating.len());
        self.free
            .iter()
            .enumerate()
            .for_each(|(k, &index)| rates[index] = free_rates[k]);
        rates.into()
    }
}

impl NodalLumpedHeatCapacities {
    pub fn total(&self) -> Quantity<HeatCapacity> {
        self.iter().copied().sum()
    }
}
