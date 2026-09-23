#[cfg(test)]
pub mod test;

mod grid;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        grid::Isosurface,
        mesh::{
            Connectivity, Mesh,
            tessellation::{D, Tessellation},
        },
    },
    math::TensorVec,
};
use std::cell::OnceCell;

impl From<Isosurface> for Tessellation {
    fn from(surface: Isosurface) -> Self {
        let mut coordinates = Coordinates::new();
        surface
            .vertices
            .iter()
            .for_each(|&vertex| coordinates.push(Coordinate::const_from(vertex)));
        let connectivities = vec![Connectivity::Triangular(surface.faces.into())];
        Tessellation::from(Mesh::from((connectivities, coordinates)))
    }
}

impl From<Mesh<D>> for Tessellation {
    fn from(mesh: Mesh<D>) -> Self {
        let normals = mesh.normals();
        Self {
            mesh,
            normals,
            bvh: OnceCell::new(),
            features: OnceCell::new(),
        }
    }
}
