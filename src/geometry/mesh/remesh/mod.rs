mod isotropic;

use crate::geometry::mesh::Mesh;

pub enum Remeshing {
    Adaptive { iterations: usize },
    Isotropic { iterations: usize },
}

impl<const D: usize> Mesh<D> {
    pub fn remesh(self, remeshing: Remeshing) -> Result<(), &'static str> {
        match remeshing {
            Remeshing::Adaptive { .. } => todo!(),
            Remeshing::Isotropic { iterations } => self.isotropic_remesh(iterations),
        }
    }
}
