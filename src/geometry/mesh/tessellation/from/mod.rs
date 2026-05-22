#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{TriangularMesh, tessellation::Tessellation},
    math::{CrossProduct, Tensor},
};

impl<const I: usize, T> From<TriangularMesh<I, T>> for Tessellation<I, T>
where
    T: Copy + Into<usize>,
{
    fn from(mesh: TriangularMesh<I, T>) -> Self {
        let normals = mesh
            .connectivity
            .iter()
            .map(|&[node_0, node_1, node_2]| {
                let node_0 = node_0.into();
                let u = &mesh.coordinates[node_1.into()] - &mesh.coordinates[node_0];
                let v = &mesh.coordinates[node_2.into()] - &mesh.coordinates[node_0];
                u.cross(v).normalized()
            })
            .collect();
        Self { mesh, normals }
    }
}
