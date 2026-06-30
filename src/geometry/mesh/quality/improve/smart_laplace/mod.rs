#[cfg(test)]
mod test;

use super::Incidence;
use crate::{
    geometry::{Coordinate, mesh::Mesh},
    math::Scalar,
};

impl<const D: usize> Mesh<D> {
    pub fn smart_laplace_smooth(&mut self, iterations: usize, scale: Scalar) {
        let number_of_nodes = self.number_of_nodes();
        let neighbors = self.node_node_connectivity().to_vec();
        let incidence = Incidence::of(self);
        let coordinates = self.coordinates.members_mut();
        for _ in 0..iterations {
            for node in 0..number_of_nodes {
                if neighbors[node].is_empty() {
                    continue;
                }
                let before = incidence.minimum_scaled_jacobian(node, coordinates);
                let original = coordinates[node].clone();
                let centroid = neighbors[node]
                    .iter()
                    .map(|&neighbor| &coordinates[neighbor])
                    .sum::<Coordinate<D>>()
                    / (neighbors[node].len() as Scalar);
                coordinates[node] += (centroid - &original) * scale;
                let after = incidence.minimum_scaled_jacobian(node, coordinates);
                if after < before {
                    coordinates[node] = original;
                }
            }
        }
    }
}
