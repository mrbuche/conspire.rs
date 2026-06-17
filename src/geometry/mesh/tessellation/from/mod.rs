#[cfg(test)]
pub mod test;

mod grid;

use crate::{
    geometry::mesh::{
        Connectivity, Mesh,
        tessellation::{D, Tessellation},
    },
    math::{CrossProduct, Tensor},
};
use std::cell::OnceCell;

impl From<Mesh<D>> for Tessellation {
    fn from(mesh: Mesh<D>) -> Self {
        let normals = mesh
            .connectivities()
            .iter()
            .map(|connectivity| {
                match connectivity {
                    Connectivity::Triangular(triangles) => {
                        triangles.iter().map(|&[node_0, node_1, node_2]| {
                            let u = &mesh.coordinates()[node_1] - &mesh.coordinates()[node_0];
                            let v = &mesh.coordinates()[node_2] - &mesh.coordinates()[node_0];
                            u.cross(v).normalized()
                        })
                    }
                    _ => panic!(),
                }
                .collect()
            })
            .collect();
        Self {
            mesh,
            normals,
            bvh: OnceCell::new(),
        }
    }
}
