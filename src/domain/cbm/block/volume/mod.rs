use super::Block;
use crate::{
    domain::{NodalReferenceCoordinates, volume::NodalVolumeElements},
    math::Quantity,
    mechanics::ReferenceCoordinate,
    units::Volume,
};

impl<C, R> NodalVolumeElements for Block<C, R> {
    fn for_each_sample(
        &self,
        reference_coordinates: &NodalReferenceCoordinates<3>,
        sample: &mut impl FnMut(usize, &ReferenceCoordinate, Quantity<Volume>),
    ) {
        self.nodes
            .iter()
            .enumerate()
            .for_each(|(index, node)| sample(index, &reference_coordinates[index], node.volume()))
    }
}
