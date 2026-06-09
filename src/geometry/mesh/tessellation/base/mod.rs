#[cfg(test)]
mod test;

use crate::geometry::{
    bvh::BoundingVolumeHierarchy,
    mesh::{
        Mesh,
        tessellation::{D, Normals, Tessellation},
    },
};

impl Tessellation {
    pub fn mesh(&self) -> &Mesh<D> {
        &self.mesh
    }
    pub fn normals(&self) -> &Normals {
        &self.normals
    }
    pub fn bvh(&self) -> &BoundingVolumeHierarchy<D> {
        self.bvh
            .get_or_init(|| BoundingVolumeHierarchy::from(&self.mesh))
    }
}
