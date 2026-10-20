use crate::{
    domain::volume::NodalVolumeElements,
    math::{Quantity, Scalar, Tensor},
    mechanics::ReferenceCoordinate,
    units::Volume,
    vem::{
        NodalReferenceCoordinates,
        block::{Block, element::mass::LumpedMassVirtualElement},
    },
};

impl<C, F, R> NodalVolumeElements for Block<C, F, R>
where
    F: LumpedMassVirtualElement,
{
    fn for_each_sample(
        &self,
        reference_coordinates: &NodalReferenceCoordinates,
        sample: &mut impl FnMut(usize, &ReferenceCoordinate, Quantity<Volume>),
    ) {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .for_each(|(element, nodes)| {
                let coordinates = Self::element_coordinates(reference_coordinates, nodes);
                let center =
                    coordinates.iter().sum::<ReferenceCoordinate>() / coordinates.len() as Scalar;
                element
                    .nodal_volumes(&coordinates)
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(volume, &node)| sample(node, &center, volume))
            })
    }
}
