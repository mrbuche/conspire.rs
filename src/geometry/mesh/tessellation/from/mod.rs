#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{TriangularMesh, tessellation::Tessellation},
    math::{CrossProduct, Tensor},
};

impl<const I: usize> From<TriangularMesh<I, usize>> for Tessellation<I, usize> {
    fn from(mesh: TriangularMesh<I, usize>) -> Self {
        let coordinates = &mesh.coordinates;
        let normals = mesh
            .connectivity
            .iter()
            .map(|&[node_0, node_1, node_2]| {
                let u = &coordinates[node_1] - &coordinates[node_0];
                let v = &coordinates[node_2] - &coordinates[node_0];
                u.cross(v).normalized()
            })
            .collect();
        Self { mesh, normals }
    }
}
