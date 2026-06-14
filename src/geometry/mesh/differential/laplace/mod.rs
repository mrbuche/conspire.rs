#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinate, Coordinates, mesh::Mesh},
    math::Scalar,
};
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub enum Weighting {
    Uniform,
    Cotangent,
}

fn edge_key(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

impl<const D: usize> Mesh<D> {
    pub fn laplacian(&self, weighting: Weighting) -> Coordinates<D> {
        let coordinates = self.coordinates();
        match weighting {
            Weighting::Uniform => self
                .node_node_connectivity()
                .iter()
                .enumerate()
                .map(|(node_a, nodes)| {
                    &coordinates[node_a]
                        - nodes
                            .iter()
                            .map(|&node_b| &coordinates[node_b])
                            .sum::<Coordinate<D>>()
                            / (nodes.len() as Scalar)
                })
                .collect(),
            Weighting::Cotangent => {
                let weights = self.cotangent_weights();
                self.node_node_connectivity()
                    .iter()
                    .enumerate()
                    .map(|(node_a, nodes)| {
                        let mut total = 0.0;
                        let displacement = nodes
                            .iter()
                            .map(|&node_b| {
                                let weight = weights[&edge_key(node_a, node_b)];
                                total += weight;
                                (&coordinates[node_a] - &coordinates[node_b]) * weight
                            })
                            .sum::<Coordinate<D>>();
                        displacement / total
                    })
                    .collect()
            }
        }
    }
    fn cotangent_weights(&self) -> HashMap<(usize, usize), Scalar> {
        let coordinates = self.coordinates();
        let mut weights = HashMap::new();
        for block in self.iter() {
            if block.number_of_nodes_per_element() == Some(3) {
                for element in block.iter() {
                    let triangle = [element[0], element[1], element[2]];
                    for local in 0..3 {
                        let i = triangle[local];
                        let j = triangle[(local + 1) % 3];
                        let k = triangle[(local + 2) % 3];
                        let u = &coordinates[i] - &coordinates[k];
                        let v = &coordinates[j] - &coordinates[k];
                        let dot = &u * &v;
                        let cross = ((&u * &u) * (&v * &v) - dot * dot).sqrt();
                        *weights.entry(edge_key(i, j)).or_insert(0.0) += dot / cross;
                    }
                }
            }
        }
        weights
    }
}
