#[cfg(test)]
mod test;

use crate::{
    domain::volume::{NodalVolumeElements, NodalVolumes},
    fem::block::{Block, element::FiniteElement},
    math::{Quantity, Tensor},
    units::Volume,
};

impl<C, F, R, H, const G: usize, const M: usize, const N: usize, const P: usize> NodalVolumeElements
    for Block<C, F, G, M, N, P, R, H>
where
    F: FiniteElement<G, M, N, P>,
{
    fn nodal_volumes_into(&self, nodal_volumes: &mut NodalVolumes) {
        let shape_functions = F::shape_functions_at_integration_points();
        self.elements().iter().zip(self.connectivity()).for_each(
            |(element, element_connectivity)| {
                element_connectivity
                    .iter()
                    .enumerate()
                    .for_each(|(a, &node)| {
                        nodal_volumes[node] += element
                            .integration_weights()
                            .iter()
                            .zip(&shape_functions)
                            .map(|(&weight, shape_functions)| weight * shape_functions[a])
                            .sum::<Quantity<Volume>>()
                    })
            },
        )
    }
}
