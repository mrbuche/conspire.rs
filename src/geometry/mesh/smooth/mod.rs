mod laplace;
mod taubin;

use crate::{
    geometry::mesh::{Mesh, differential::laplace::Weighting},
    math::Scalar,
};

pub enum Smoothing {
    Laplace {
        iterations: usize,
        scale: Scalar,
        weighting: Weighting,
        preserve_boundary: bool,
    },
    Taubin {
        iterations: usize,
        pass_band: Scalar,
        scale: Scalar,
        weighting: Weighting,
        preserve_boundary: bool,
    },
}

impl<const D: usize> Mesh<D> {
    pub fn smooth(&mut self, smoothing: Smoothing) {
        match smoothing {
            Smoothing::Laplace {
                iterations,
                scale,
                weighting,
                preserve_boundary,
            } => self.laplace_smooth(iterations, scale, weighting, preserve_boundary),
            Smoothing::Taubin {
                iterations,
                pass_band,
                scale,
                weighting,
                preserve_boundary,
            } => self.taubin_smooth(iterations, pass_band, scale, weighting, preserve_boundary),
        }
    }
    pub(crate) fn boundary_preserving_adjacency(&self) -> Vec<Vec<usize>> {
        let mut is_boundary = vec![false; self.number_of_nodes()];
        self.exterior_faces()
            .iter()
            .flatten()
            .for_each(|&node| is_boundary[node] = true);
        let mut adjacency: Vec<Vec<usize>> = self.node_node_connectivity().to_vec();
        adjacency
            .iter_mut()
            .enumerate()
            .filter(|(node, _)| is_boundary[*node])
            .for_each(|(_, neighbors)| neighbors.retain(|&other| is_boundary[other]));
        adjacency
    }
}
