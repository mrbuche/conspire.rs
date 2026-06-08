mod adaptive;
mod isotropic;
mod triangles;

use crate::{geometry::mesh::Mesh, math::Scalar};

const D: usize = 3;

pub enum Remeshing {
    Adaptive {
        iterations: usize,
        tolerance: Scalar,
        minimum: Scalar,
        maximum: Scalar,
        gradation: Scalar,
    },
    Isotropic {
        iterations: usize,
        length: Option<Scalar>,
    },
}

impl Mesh<D> {
    pub fn remesh(self, remeshing: Remeshing) -> Result<Self, &'static str> {
        match remeshing {
            Remeshing::Adaptive {
                iterations,
                tolerance,
                minimum,
                maximum,
                gradation,
            } => self.adaptive_remesh(iterations, tolerance, minimum, maximum, gradation),
            Remeshing::Isotropic { iterations, length } => {
                self.isotropic_remesh(iterations, length)
            }
        }
    }
}
