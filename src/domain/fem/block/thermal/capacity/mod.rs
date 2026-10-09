use crate::{
    fem::{
        block::{
            Block,
            element::thermal::capacity::{
                HeatCapacityFiniteElement, IntegrationHeatCapacities,
                LumpedHeatCapacityFiniteElement,
            },
        },
        thermal::capacity::{
            ConsistentHeatCapacityElements, LumpedHeatCapacityElements, NodalHeatCapacities,
            NodalLumpedHeatCapacities,
        },
    },
    math::{Quantity, TensorListVec},
    units::VolumetricHeatCapacity,
};

pub type ElementHeatCapacities<const G: usize> = TensorListVec<Quantity<VolumetricHeatCapacity>, G>;

pub trait HeatCapacities<const G: usize> {
    fn at(&self, element: usize) -> IntegrationHeatCapacities<G>;
}

impl<const G: usize> HeatCapacities<G> for ElementHeatCapacities<G> {
    fn at(&self, element: usize) -> IntegrationHeatCapacities<G> {
        self[element].clone()
    }
}

impl<const G: usize> HeatCapacities<G> for Quantity<VolumetricHeatCapacity> {
    fn at(&self, _element: usize) -> IntegrationHeatCapacities<G> {
        [*self; G].into()
    }
}

impl<C, F, R, const G: usize, const M: usize, const N: usize, const P: usize>
    Block<C, F, G, M, N, P, R>
{
    pub fn with_heat_capacity<S>(self, heat_capacity: S) -> Block<C, F, G, M, N, P, S>
    where
        S: HeatCapacities<G>,
    {
        self.with_density(heat_capacity)
    }
}

impl<C, F, R, const G: usize, const M: usize, const N: usize, const P: usize>
    ConsistentHeatCapacityElements for Block<C, F, G, M, N, P, R>
where
    F: HeatCapacityFiniteElement<G, M, N, P>,
    R: HeatCapacities<G>,
{
    fn nodal_heat_capacities_into(&self, nodal_heat_capacities: &mut NodalHeatCapacities) {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .enumerate()
            .for_each(|(element_index, (element, element_connectivity))| {
                element
                    .nodal_heat_capacities(&self.density().at(element_index))
                    .into_iter()
                    .zip(element_connectivity)
                    .for_each(|(object, &node_a)| {
                        object.into_iter().zip(element_connectivity).for_each(
                            |(nodal_heat_capacity, &node_b)| {
                                nodal_heat_capacities[node_a][node_b] += nodal_heat_capacity
                            },
                        )
                    })
            })
    }
}

impl<C, F, R, const G: usize, const M: usize, const N: usize, const P: usize>
    LumpedHeatCapacityElements for Block<C, F, G, M, N, P, R>
where
    F: LumpedHeatCapacityFiniteElement<G, M, N, P>,
    R: HeatCapacities<G>,
{
    fn nodal_lumped_heat_capacities_into(
        &self,
        nodal_lumped_heat_capacities: &mut NodalLumpedHeatCapacities,
    ) {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .enumerate()
            .for_each(|(element_index, (element, element_connectivity))| {
                element
                    .nodal_lumped_heat_capacities(&self.density().at(element_index))
                    .into_iter()
                    .zip(element_connectivity)
                    .for_each(|(nodal_heat_capacity, &node)| {
                        nodal_lumped_heat_capacities[node] += nodal_heat_capacity
                    })
            })
    }
}
