#[cfg(test)]
mod test;

use crate::{
    domain::volume::NodalVolumeElements,
    fem::{
        NodalReferenceCoordinates,
        block::{Block, VolumeFiniteElement},
    },
    math::{Quantity, Tensor},
    mechanics::ReferenceCoordinate,
    units::Volume,
};

impl<C, F, R, H, const G: usize, const N: usize> NodalVolumeElements for Block<C, F, G, 3, N, R, H>
where
    F: VolumeFiniteElement<G, 3, N>,
{
    fn for_each_sample(
        &self,
        reference_coordinates: &NodalReferenceCoordinates<3>,
        sample: &mut impl FnMut(usize, &ReferenceCoordinate, Quantity<Volume>),
    ) {
        let shape_functions = F::shape_functions_at_integration_points();
        self.elements().iter().zip(self.connectivity()).for_each(
            |(element, element_connectivity)| {
                element
                    .integration_weights()
                    .iter()
                    .zip(&shape_functions)
                    .for_each(|(&weight, shape_functions)| {
                        let position = element_connectivity
                            .iter()
                            .zip(shape_functions.iter())
                            .map(|(&node, &shape_function)| {
                                &reference_coordinates[node] * shape_function
                            })
                            .sum::<ReferenceCoordinate>();
                        element_connectivity
                            .iter()
                            .zip(shape_functions.iter())
                            .for_each(|(&node, &shape_function)| {
                                sample(node, &position, weight * shape_function)
                            })
                    })
            },
        )
    }
}
