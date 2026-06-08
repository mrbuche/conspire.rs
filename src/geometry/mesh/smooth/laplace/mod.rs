#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Mesh, differential::laplace::Weighting},
    math::Scalar,
};

impl<const D: usize> Mesh<D> {
    pub fn laplace_smooth(&mut self, iterations: usize, scale: Scalar, weighting: Weighting) {
        for _ in 0..iterations {
            let laplacian = self.laplacian(weighting);
            self.coordinates
                .iter_mut()
                .zip(laplacian)
                .for_each(|(x, u)| *x -= u * scale)
        }
    }
}
