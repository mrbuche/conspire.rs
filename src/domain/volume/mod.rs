use crate::{
    domain::{Blocks, ElementModel, Model, NodalReferenceCoordinates, block::element::Elements},
    math::{Quantity, QuantityVector, Tensor},
    mechanics::ReferenceCoordinate,
    units::Volume,
};

/// The volume that each node has of the elements around it.
pub type NodalVolumes = QuantityVector<Volume>;

/// Elements that share their volume among their nodes at the points where they sample it.
pub trait NodalVolumeElements
where
    Self: Elements,
{
    /// Calls `sample` with a node, a reference position and a volume for every sample, where the
    /// volumes of the samples of a node add up to the volume of the node.
    fn for_each_sample(
        &self,
        reference_coordinates: &NodalReferenceCoordinates<3>,
        sample: &mut impl FnMut(usize, &ReferenceCoordinate, Quantity<Volume>),
    );
    fn nodal_volumes_into(
        &self,
        reference_coordinates: &NodalReferenceCoordinates<3>,
        nodal_volumes: &mut NodalVolumes,
    ) {
        self.for_each_sample(reference_coordinates, &mut |node, _, volume| {
            nodal_volumes[node] += volume
        })
    }
}

impl<B1, B2> NodalVolumeElements for Blocks<B1, B2>
where
    B1: NodalVolumeElements,
    B2: NodalVolumeElements,
{
    fn for_each_sample(
        &self,
        reference_coordinates: &NodalReferenceCoordinates<3>,
        sample: &mut impl FnMut(usize, &ReferenceCoordinate, Quantity<Volume>),
    ) {
        self.0.for_each_sample(reference_coordinates, &mut *sample);
        self.1.for_each_sample(reference_coordinates, &mut *sample)
    }
}

impl<B, const D: usize> NodalVolumeElements for Model<B, D>
where
    B: NodalVolumeElements,
{
    fn for_each_sample(
        &self,
        reference_coordinates: &NodalReferenceCoordinates<3>,
        sample: &mut impl FnMut(usize, &ReferenceCoordinate, Quantity<Volume>),
    ) {
        self.blocks.for_each_sample(reference_coordinates, sample)
    }
}

impl<B> Model<B, 3>
where
    B: NodalVolumeElements,
{
    /// The volume that each node has of the elements around it, whose sum is the volume.
    pub fn nodal_volumes(&self) -> NodalVolumes {
        let mut nodal_volumes = NodalVolumes::zero(self.coordinates().len());
        self.nodal_volumes_into(self.coordinates(), &mut nodal_volumes);
        nodal_volumes
    }
}

impl NodalVolumes {
    /// The volume of the elements, which is the sum over their nodes.
    pub fn volume(&self) -> Quantity<Volume> {
        self.iter().copied().sum()
    }
}
