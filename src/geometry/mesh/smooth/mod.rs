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
        preserve_interfaces: bool,
    },
    Taubin {
        iterations: usize,
        pass_band: Scalar,
        scale: Scalar,
        weighting: Weighting,
        preserve_boundary: bool,
        preserve_interfaces: bool,
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
                preserve_interfaces,
            } => self.laplace_smooth(
                iterations,
                scale,
                weighting,
                preserve_boundary,
                preserve_interfaces,
            ),
            Smoothing::Taubin {
                iterations,
                pass_band,
                scale,
                weighting,
                preserve_boundary,
                preserve_interfaces,
            } => self.taubin_smooth(
                iterations,
                pass_band,
                scale,
                weighting,
                preserve_boundary,
                preserve_interfaces,
            ),
        }
    }
    fn constrained_adjacency(
        &self,
        preserve_boundary: bool,
        preserve_interfaces: bool,
    ) -> Vec<Vec<usize>> {
        let mut constrained = vec![false; self.number_of_nodes()];
        if preserve_boundary {
            self.exterior_faces()
                .iter()
                .flatten()
                .for_each(|&node| constrained[node] = true);
        }
        if preserve_interfaces {
            self.mark_interface_nodes(&mut constrained);
        }
        let mut adjacency: Vec<Vec<usize>> = self.node_node_connectivity().to_vec();
        adjacency
            .iter_mut()
            .enumerate()
            .filter(|(node, _)| constrained[*node])
            .for_each(|(_, neighbors)| neighbors.retain(|&other| constrained[other]));
        adjacency
    }
    fn mark_interface_nodes(&self, constrained: &mut [bool]) {
        let mut element_block = Vec::with_capacity(self.number_of_elements());
        for (block, connectivity) in self.connectivities().iter().enumerate() {
            element_block.resize(
                element_block.len() + connectivity.number_of_elements(),
                block,
            );
        }
        self.node_element_connectivity()
            .iter()
            .enumerate()
            .for_each(|(node, elements)| {
                let mut blocks = elements.iter().map(|&element| element_block[element]);
                if let Some(first) = blocks.next()
                    && blocks.any(|block| block != first)
                {
                    constrained[node] = true;
                }
            });
    }
}
