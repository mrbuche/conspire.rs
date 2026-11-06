#[cfg(test)]
pub mod test;

mod grid;

use crate::geometry::{
    Coordinates,
    grid::Isosurface,
    mesh::{
        Connectivity, Mesh,
        tessellation::{D, Tessellation},
    },
};
use core::cell::OnceCell;

impl From<Isosurface> for Tessellation {
    fn from(surface: Isosurface) -> Self {
        (surface.faces, surface.vertices).into()
    }
}

impl<C> From<(Vec<[usize; 3]>, C)> for Tessellation
where
    C: Into<Coordinates<D>>,
{
    fn from((triangles, coordinates): (Vec<[usize; 3]>, C)) -> Self {
        Mesh::from((
            vec![Connectivity::Triangular(triangles.into())],
            coordinates.into(),
        ))
        .into()
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
