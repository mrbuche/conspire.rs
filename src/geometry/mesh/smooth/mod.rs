mod laplace;
mod taubin;

use crate::{
    geometry::mesh::{Mesh, operator::laplace::Weighting},
    math::Scalar,
};

pub enum Smoothing {
    Laplace {
        iterations: usize,
        scale: Scalar,
        weighting: Weighting,
    },
    Taubin {
        iterations: usize,
        pass_band: Scalar,
        scale: Scalar,
        weighting: Weighting,
    },
}

impl<const D: usize> Mesh<D> {
    pub fn smooth(&mut self, smoothing: Smoothing) {
        match smoothing {
            Smoothing::Laplace {
                iterations,
                scale,
                weighting,
            } => self.laplace_smooth(iterations, scale, weighting),
            Smoothing::Taubin {
                iterations,
                pass_band,
                scale,
                weighting,
            } => self.taubin_smooth(iterations, pass_band, scale, weighting),
        }
    }
}
