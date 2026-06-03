#[cfg(test)]
mod test;

use crate::geometry::{
    bvh::{BoundingVolumeHierarchy, primitive::Primitives},
    mesh::{Mesh, Tessellation},
};

const LEAF_SIZE: usize = 4;

impl<const D: usize> From<Primitives<D>> for BoundingVolumeHierarchy<D> {
    fn from(mut primitives: Primitives<D>) -> Self {
        let mut bvh = Self {
            items: Vec::new(),
            nodes: Vec::new(),
        };
        bvh.build_node(&mut primitives, LEAF_SIZE);
        bvh
    }
}

impl<const D: usize> From<&Mesh<D>> for BoundingVolumeHierarchy<D> {
    fn from(mesh: &Mesh<D>) -> Self {
        Primitives::from(mesh).into()
    }
}

impl From<&Tessellation> for BoundingVolumeHierarchy<3> {
    fn from(tessellation: &Tessellation) -> Self {
        tessellation.mesh().into()
    }
}
