#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Mesh, operator::laplace::Weighting},
    math::Scalar,
};

impl<const D: usize> Mesh<D> {
    pub fn laplace_smooth(&mut self, iterations: usize, scale: Scalar) {
        for _ in 0..iterations {
            let laplacian = self.laplacian(Weighting::Uniform);
            self.coordinates
                .iter_mut()
                .zip(laplacian)
                .for_each(|(x, u)| *x -= u * scale)
        }
    }
}
