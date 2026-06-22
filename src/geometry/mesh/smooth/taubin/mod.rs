#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Mesh, differential::laplace::Weighting},
    math::Scalar,
};

impl<const D: usize> Mesh<D> {
    pub fn taubin_smooth(
        &mut self,
        iterations: usize,
        pass_band: Scalar,
        scale: Scalar,
        weighting: Weighting,
        preserve_boundary: bool,
        preserve_interfaces: bool,
    ) {
        let scale_deflate = scale;
        let scale_inflate = scale / (pass_band * scale - 1.0);
        let adjacency = (preserve_boundary || preserve_interfaces)
            .then(|| self.constrained_adjacency(preserve_boundary, preserve_interfaces));
        for iteration in 0..iterations {
            let laplacian = match &adjacency {
                Some(adjacency) => self.laplacian_over(adjacency, weighting),
                None => self.laplacian(weighting),
            };
            let scale = if scale_inflate < 0.0 && iteration % 2 == 1 {
                scale_inflate
            } else {
                scale_deflate
            };
            self.coordinates
                .iter_mut()
                .zip(laplacian)
                .for_each(|(x, u)| *x -= u * scale)
        }
    }
}
