#[cfg(test)]
mod test;

use crate::geometry::{mesh::Mesh, segmentation::Segmentation};

impl Mesh<3> {
    pub fn from_segmentation<T>(segmentation: Segmentation<3, T>, remove: Option<&[T]>) -> Self
    where
        T: Copy + PartialEq + Into<usize>,
    {
        let (grid, scale, translate) = segmentation.into_parts();
        Self::from_voxels_embedded(grid, remove, &scale, &translate)
    }
}
