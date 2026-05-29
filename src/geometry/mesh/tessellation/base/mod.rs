#[cfg(test)]
mod test;

use crate::geometry::mesh::{
    Mesh,
    tessellation::{D, Normals, Tessellation},
};

impl Tessellation {
    pub fn mesh(&self) -> &Mesh<D> {
        &self.mesh
    }
    pub fn normals(&self) -> &Normals {
        &self.normals
    }
}
