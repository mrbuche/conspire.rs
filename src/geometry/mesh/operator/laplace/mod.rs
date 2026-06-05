#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinate, Coordinates, mesh::Mesh},
    math::{Scalar, TensorArray},
};

impl<const D: usize> Mesh<D> {
    pub fn laplacian(&self) -> Coordinates<D> {
        let coordinates = self.coordinates();
        self.node_node_connectivity()
            .iter()
            .enumerate()
            .map(|(node_a, nodes)| {
                if nodes.is_empty() {
                    Coordinate::zero()
                } else {
                    &coordinates[node_a]
                        - nodes
                            .iter()
                            .map(|&node_b| &coordinates[node_b])
                            .sum::<Coordinate<D>>()
                            / (nodes.len() as Scalar)
                }
            })
            .collect()
    }
}
