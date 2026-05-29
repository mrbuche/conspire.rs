#[cfg(test)]
mod test;

use crate::geometry::{
    bvh::{BoundingVolumeHierarchy, primitive::Primitives},
    mesh::{Mesh, Tessellation},
};

impl<const D: usize> From<Primitives<D>> for BoundingVolumeHierarchy<D> {
    fn from(mut primitives: Primitives<D>) -> Self {
        let mut bvh = Self {
            items: Vec::new(),
            nodes: Vec::new(),
        };
        let leaf_size = 4;
        bvh.build_node(&mut primitives, leaf_size);
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
        BoundingVolumeHierarchy::<3>::from(tessellation.mesh())
    }
}
