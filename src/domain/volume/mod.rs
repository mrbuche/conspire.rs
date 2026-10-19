use crate::{
    domain::{Blocks, ElementModel, Model, block::element::Elements},
    math::{Quantity, QuantityVector, Tensor},
    units::Volume,
};

/// The volume that each node has of the elements around it.
pub type NodalVolumes = QuantityVector<Volume>;

/// Elements that share their volume among their nodes, as the lumped mass does its mass.
pub trait NodalVolumeElements
where
    Self: Elements,
{
    fn nodal_volumes_into(&self, nodal_volumes: &mut NodalVolumes);
}

impl<B1, B2> NodalVolumeElements for Blocks<B1, B2>
where
    B1: NodalVolumeElements,
    B2: NodalVolumeElements,
{
    fn nodal_volumes_into(&self, nodal_volumes: &mut NodalVolumes) {
        self.0.nodal_volumes_into(nodal_volumes);
        self.1.nodal_volumes_into(nodal_volumes)
    }
}

impl<B, const D: usize> NodalVolumeElements for Model<B, D>
where
    B: NodalVolumeElements,
{
    fn nodal_volumes_into(&self, nodal_volumes: &mut NodalVolumes) {
        self.blocks.nodal_volumes_into(nodal_volumes)
    }
}

impl<B, const D: usize> Model<B, D>
where
    B: NodalVolumeElements,
{
    /// The volume that each node has of the elements around it, whose sum is the volume.
    pub fn nodal_volumes(&self) -> NodalVolumes {
        let mut nodal_volumes = NodalVolumes::zero(self.coordinates().len());
        self.nodal_volumes_into(&mut nodal_volumes);
        nodal_volumes
    }
}

impl NodalVolumes {
    /// The volume of the elements, which is the sum over their nodes.
    pub fn volume(&self) -> Quantity<Volume> {
        self.iter().copied().sum()
    }
}
