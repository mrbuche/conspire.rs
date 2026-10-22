use crate::{
    fem::block::element::{
        Element, FiniteElement,
        mass::{ConsistentMass, RowSumLumping},
    },
    math::{Quantity, Tensor, TensorList},
    units::{HeatCapacity, VolumetricHeatCapacity},
};

pub type IntegrationHeatCapacities<const G: usize> =
    TensorList<Quantity<VolumetricHeatCapacity>, G>;
pub type ElementNodalHeatCapacities<const D: usize> =
    TensorList<TensorList<Quantity<HeatCapacity>, D>, D>;
pub type ElementNodalLumpedHeatCapacities<const D: usize> = TensorList<Quantity<HeatCapacity>, D>;

pub trait HeatCapacityFiniteElement<const G: usize, const M: usize, const N: usize>
where
    Self: FiniteElement<G, M, N, N> + ConsistentMass,
{
    fn nodal_heat_capacities(
        &self,
        capacities: &IntegrationHeatCapacities<G>,
    ) -> ElementNodalHeatCapacities<N>;
}

pub trait LumpedHeatCapacityFiniteElement<const G: usize, const M: usize, const N: usize>
where
    Self: FiniteElement<G, M, N, N> + RowSumLumping,
{
    fn nodal_lumped_heat_capacities(
        &self,
        capacities: &IntegrationHeatCapacities<G>,
    ) -> ElementNodalLumpedHeatCapacities<N>;
}

impl<const G: usize, const M: usize, const N: usize, const O: usize>
    HeatCapacityFiniteElement<G, M, N> for Element<3, G, N, O>
where
    Self: FiniteElement<G, M, N, N> + ConsistentMass,
{
    fn nodal_heat_capacities(
        &self,
        capacities: &IntegrationHeatCapacities<G>,
    ) -> ElementNodalHeatCapacities<N> {
        Self::shape_functions_at_integration_points()
            .iter()
            .zip(capacities)
            .zip(self.integration_weights())
            .map(|((shape_functions, capacity), integration_weight)| {
                let capacity = capacity * integration_weight;
                shape_functions
                    .iter()
                    .map(|shape_function_a| {
                        shape_functions
                            .iter()
                            .map(|shape_function_b| {
                                capacity * (shape_function_a * shape_function_b)
                            })
                            .collect()
                    })
                    .collect()
            })
            .sum()
    }
}

impl<const G: usize, const M: usize, const N: usize, const O: usize>
    LumpedHeatCapacityFiniteElement<G, M, N> for Element<3, G, N, O>
where
    Self: FiniteElement<G, M, N, N> + RowSumLumping,
{
    fn nodal_lumped_heat_capacities(
        &self,
        capacities: &IntegrationHeatCapacities<G>,
    ) -> ElementNodalLumpedHeatCapacities<N> {
        Self::shape_functions_at_integration_points()
            .iter()
            .zip(capacities)
            .zip(self.integration_weights())
            .map(|((shape_functions, capacity), integration_weight)| {
                let capacity = capacity * integration_weight;
                shape_functions
                    .iter()
                    .map(|shape_function| capacity * shape_function)
                    .collect()
            })
            .sum()
    }
}
