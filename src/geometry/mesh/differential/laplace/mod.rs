#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinate, Coordinates, mesh::Mesh},
    math::Scalar,
};

#[derive(Clone, Copy)]
pub enum Weighting {
    Uniform,
    Cotangent,
}

impl<const D: usize> Mesh<D> {
    pub fn laplacian(&self, weighting: Weighting) -> Coordinates<D> {
        let coordinates = self.coordinates();
        self.node_node_connectivity()
            .iter()
            .enumerate()
            .map(|(node_a, nodes)| match weighting {
                Weighting::Uniform => {
                    &coordinates[node_a]
                        - nodes
                            .iter()
                            .map(|&node_b| &coordinates[node_b])
                            .sum::<Coordinate<D>>()
                            / (nodes.len() as Scalar)
                }
                Weighting::Cotangent => todo!(),
            })
            .collect()
    }
}
