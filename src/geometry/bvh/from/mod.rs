#[cfg(test)]
mod test;

use crate::geometry::{
    bbox::BoundingBoxes,
    bvh::{
        BoundingVolumeHierarchy,
        primitive::{Primitive, Primitives},
    },
    mesh::{Mesh, Tessellation},
};

const LEAF_SIZE: usize = 4;

/// Builds a hierarchy over whatever the boxes bound.
///
/// Items with nothing but a box to them, as a solid described by where it is
/// rather than by a mesh of it, are split on their boxes' centers for want of a
/// truer centroid to prefer.
impl<const D: usize> From<BoundingBoxes<D>> for BoundingVolumeHierarchy<D> {
    fn from(bounding_boxes: BoundingBoxes<D>) -> Self {
        bounding_boxes
            .into_iter()
            .enumerate()
            .map(|(index, bounding_box)| {
                let centroid = bounding_box.center();
                Primitive::new(bounding_box, centroid, index)
            })
            .collect::<Primitives<D>>()
            .into()
    }
}

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
