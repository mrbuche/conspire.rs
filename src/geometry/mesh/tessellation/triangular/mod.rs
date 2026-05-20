#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{TriangularMesh, tessellation::Tessellation},
    math::Tensor,
};

pub type TriangularTessellation<const I: usize, T> = Tessellation<3, I, 2, Vec<[T; 3]>>;

impl<const I: usize> From<TriangularMesh<I, usize>> for TriangularTessellation<I, usize> {
    fn from(mesh: TriangularMesh<I, usize>) -> Self {
        let coordinates = &mesh.coordinates;
        let normals = mesh
            .connectivity
            .iter()
            .map(|&[node_0, node_1, node_2]| {
                let u = &coordinates[node_1] - &coordinates[node_0];
                let v = &coordinates[node_2] - &coordinates[node_0];
                u.cross(&v).normalized() // make a Cross trait so can add version that consumes both
            })
            .collect();
        Self { mesh, normals }
    }
}
