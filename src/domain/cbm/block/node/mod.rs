pub mod solid;

use crate::{
    domain::NodalReferenceCoordinates,
    fem::block::element::{
        ElementNodalReferenceCoordinates, FiniteElement, GradientVectors, linear::Tetrahedron,
    },
    geometry::mesh::PrimitiveConnectivity,
    math::{Quantity, Reference, Tensor, TensorRank1, TensorRank1List, TensorRank1Vec},
    units::{ReciprocalLength, UnitMul, Volume},
};
use std::collections::HashMap;

/// A reference-configuration gradient vector, `\zeta_{ip}` in the source paper.
pub(crate) type BondGradientVector = TensorRank1<3, Reference, ReciprocalLength>;
/// The intermediate, unnormalized accumulation of a bond gradient vector across
/// the incident tetrahedra, before dividing through by the particle's volume.
type UnnormalizedBondGradientVector =
    TensorRank1<3, Reference, <ReciprocalLength as UnitMul<Volume>>::Output>;

/// A particle: its reference volume and the bond gradient vectors — including
/// its own self term — of the neighbors spanning its bond neighborhood, kept
/// parallel to (and in the same order as) [`Self::neighbors`].
pub(crate) struct Node {
    volume: Quantity<Volume>,
    neighbors: Vec<usize>,
    gradient_vectors: Vec<BondGradientVector>,
}

impl Node {
    pub(crate) fn neighbors(&self) -> &[usize] {
        &self.neighbors
    }
    pub(crate) fn gradient_vectors(&self) -> &[BondGradientVector] {
        &self.gradient_vectors
    }
    fn element_coordinates<const D: usize, I, U>(
        coordinates: &TensorRank1Vec<D, I, U>,
        nodes: &[usize; 4],
    ) -> TensorRank1List<D, I, 4, U> {
        nodes
            .iter()
            .map(|&node| coordinates[node].clone())
            .collect()
    }
    fn accumulate_bonds(
        connectivity: &PrimitiveConnectivity<3, 4>,
        elements: &[Tetrahedron],
    ) -> HashMap<(usize, usize), UnnormalizedBondGradientVector> {
        let mut bonds: HashMap<(usize, usize), UnnormalizedBondGradientVector> = HashMap::new();
        connectivity
            .iter()
            .zip(elements.iter())
            .for_each(|(nodes, element)| {
                let quarter_volume = element.volume() / 4.0;
                let gradient_vectors: &GradientVectors<3, 1, 4> = element.gradient_vectors();
                nodes.iter().for_each(|&node_a| {
                    nodes.iter().zip(gradient_vectors[0].iter()).for_each(
                        |(&node_b, gradient_vector_b)| {
                            if node_a != node_b {
                                let contribution = gradient_vector_b * quarter_volume;
                                bonds
                                    .entry((node_a, node_b))
                                    .and_modify(|gradient_vector| *gradient_vector += &contribution)
                                    .or_insert(contribution);
                            }
                        },
                    )
                })
            });
        bonds
    }
    /// Builds one [`Node`] per particle from the reference-configuration tet
    /// connectivity and coordinates.
    pub(crate) fn vec_from(
        connectivity: &PrimitiveConnectivity<3, 4>,
        reference_coordinates: &NodalReferenceCoordinates<3>,
    ) -> Vec<Self> {
        let elements: Vec<Tetrahedron> = connectivity
            .iter()
            .map(|nodes| -> ElementNodalReferenceCoordinates<4> {
                Self::element_coordinates(reference_coordinates, nodes)
            })
            .map(Tetrahedron::from)
            .collect();
        let mut volumes = vec![Quantity::<Volume>::new(0.0); reference_coordinates.len()];
        connectivity
            .iter()
            .zip(elements.iter())
            .for_each(|(nodes, element)| {
                let quarter_volume = element.volume() / 4.0;
                nodes
                    .iter()
                    .for_each(|&node| volumes[node] += &quarter_volume)
            });
        let bonds = Self::accumulate_bonds(connectivity, &elements);
        let mut unnormalized: Vec<Vec<(usize, UnnormalizedBondGradientVector)>> =
            vec![Vec::new(); volumes.len()];
        bonds
            .into_iter()
            .for_each(|((node_a, node_b), bond)| unnormalized[node_a].push((node_b, bond)));
        unnormalized
            .iter_mut()
            .enumerate()
            .for_each(|(node, bonds)| {
                let self_gradient_vector = -bonds
                    .iter()
                    .map(|(_, gradient_vector)| gradient_vector)
                    .sum::<UnnormalizedBondGradientVector>();
                bonds.push((node, self_gradient_vector));
            });
        unnormalized
            .into_iter()
            .zip(volumes)
            .map(|(bonds, volume)| {
                let (neighbors, gradient_vectors) = bonds
                    .into_iter()
                    .map(|(node, bond)| (node, bond / volume))
                    .unzip();
                Node {
                    volume,
                    neighbors,
                    gradient_vectors,
                }
            })
            .collect()
    }
}
