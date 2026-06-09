#[cfg(test)]
mod test;

use super::super::metrics::{self, Kind};
use crate::{
    geometry::{Coordinate, Coordinates, mesh::Mesh},
    math::Scalar,
};

impl<const D: usize> Mesh<D> {
    pub fn smart_laplace_smooth(&mut self, iterations: usize, scale: Scalar) {
        let number_of_nodes = self.number_of_nodes();
        let neighbors = self.node_node_connectivity().to_vec();
        let node_elements = self.node_element_connectivity().to_vec();
        let element_nodes: Vec<Vec<usize>> = self
            .iter()
            .flat_map(|block| block.iter().map(<[usize]>::to_vec))
            .collect();
        let element_kinds: Vec<Kind> = self
            .iter()
            .flat_map(|block| {
                let kind = Kind::of(block).expect("unsupported element type");
                block.iter().map(move |_| kind)
            })
            .collect();
        let mut coordinates = self.coordinates().clone();
        for _ in 0..iterations {
            for node in 0..number_of_nodes {
                if neighbors[node].is_empty() {
                    continue;
                }
                let before = incident_quality(
                    node,
                    &node_elements,
                    &element_nodes,
                    &element_kinds,
                    &coordinates,
                );
                let original = coordinates[node].clone();
                let centroid = neighbors[node]
                    .iter()
                    .map(|&neighbor| &coordinates[neighbor])
                    .sum::<Coordinate<D>>()
                    / (neighbors[node].len() as Scalar);
                coordinates[node] += (centroid - &original) * scale;
                let after = incident_quality(
                    node,
                    &node_elements,
                    &element_nodes,
                    &element_kinds,
                    &coordinates,
                );
                if after < before {
                    coordinates[node] = original;
                }
            }
        }
        self.coordinates
            .iter_mut()
            .zip(coordinates)
            .for_each(|(coordinate, smoothed)| *coordinate = smoothed);
    }
}

fn incident_quality<const D: usize>(
    node: usize,
    node_elements: &[Vec<usize>],
    element_nodes: &[Vec<usize>],
    element_kinds: &[Kind],
    coordinates: &Coordinates<D>,
) -> Scalar {
    node_elements[node]
        .iter()
        .map(|&element| {
            metrics::minimum_scaled_jacobian(
                element_kinds[element],
                &element_nodes[element],
                coordinates,
            )
        })
        .fold(Scalar::INFINITY, Scalar::min)
}
