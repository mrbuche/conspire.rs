mod laplace;
mod taubin;

use crate::{geometry::mesh::Mesh, math::Scalar};

pub enum Smoothing {
    Laplace {
        iterations: usize,
        scale: Scalar,
    },
    Taubin {
        iterations: usize,
        pass_band: Scalar,
        scale: Scalar,
    },
}

impl<const D: usize> Mesh<D> {
    pub fn smooth(&mut self, smoothing: Smoothing) {
        match smoothing {
            Smoothing::Laplace { iterations, scale } => self.laplace_smooth(iterations, scale),
            Smoothing::Taubin {
                iterations,
                pass_band,
                scale,
            } => self.taubin_smooth(iterations, pass_band, scale),
        }
    }
}
