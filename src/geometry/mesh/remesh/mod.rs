mod adaptive;
mod isotropic;

use crate::{geometry::mesh::Mesh, math::Scalar};

pub enum Remeshing {
    Adaptive {
        iterations: usize,
    },
    Isotropic {
        iterations: usize,
        length: Option<Scalar>,
    },
}

impl<const D: usize> Mesh<D> {
    pub fn remesh(self, remeshing: Remeshing) -> Result<Self, &'static str> {
        match remeshing {
            Remeshing::Adaptive { .. } => todo!(),
            Remeshing::Isotropic { iterations, length } => {
                self.isotropic_remesh(iterations, length)
            }
        }
    }
}
