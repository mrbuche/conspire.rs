#[cfg(test)]
mod test;

use crate::geometry::{bvh::BoundingVolumeHierarchy, mesh::Mesh};

impl<const D: usize, const I: usize, const M: usize, T> From<Mesh<D, I, M, T>>
    for BoundingVolumeHierarchy<D, I, T>
{
    fn from(mesh: Mesh<D, I, M, T>) -> Self {
        todo!(
            "Need centroid and bounding box for each element of the mesh, implement method that does both in mesh."
        )
    }
}
