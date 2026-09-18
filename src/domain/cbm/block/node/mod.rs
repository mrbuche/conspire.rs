pub mod solid;
#[cfg(test)]
mod test;
mod tetrahedron;

use crate::{
    domain::NodalReferenceCoordinates,
    geometry::mesh::PrimitiveConnectivity,
    math::{
        CrossProduct, Quantity, Reference, Scalar, Tensor, TensorRank1, TensorRank1List,
        TensorRank1Vec,
    },
    units::{ReciprocalLength, UnitMul, Volume},
};
use std::collections::HashMap;
use tetrahedron::{ElementNodalReferenceCoordinates, GradientVectors, Tetrahedron};

/// How a tetrahedron's volume is split among its 4 vertices when building
/// each particle's tributary volume and bond gradient vectors.
#[derive(Clone, Copy, Debug, Default)]
pub enum Weighting {
    /// The source paper's scheme: each vertex gets an equal 1/4 share.
    #[default]
    Uniform,
    /// Each vertex gets a share proportional to its solid angle within the
    /// tetrahedron (Van Oosterom-Strackee formula), normalized to sum to 1.
    SolidAngle,
}

impl Weighting {
    fn weights(&self, coordinates: &ElementNodalReferenceCoordinates) -> [Scalar; 4] {
        match self {
            Self::Uniform => [0.25; 4],
            Self::SolidAngle => solid_angle_weights(coordinates),
        }
    }
}

fn solid_angle_weights(coordinates: &ElementNodalReferenceCoordinates) -> [Scalar; 4] {
    let solid_angle = |apex: usize, others: [usize; 3]| -> Scalar {
        let a = &coordinates[others[0]] - &coordinates[apex];
        let b = &coordinates[others[1]] - &coordinates[apex];
        let c = &coordinates[others[2]] - &coordinates[apex];
        let numerator = (&a * &b.cross(&c)).value().abs();
        let (norm_a, norm_b, norm_c) = (a.norm().value(), b.norm().value(), c.norm().value());
        let denominator = norm_a * norm_b * norm_c
            + (&a * &b).value() * norm_c
            + (&a * &c).value() * norm_b
            + (&b * &c).value() * norm_a;
        2.0 * numerator.atan2(denominator)
    };
    let angles = [
        solid_angle(0, [1, 2, 3]),
        solid_angle(1, [0, 2, 3]),
        solid_angle(2, [0, 1, 3]),
        solid_angle(3, [0, 1, 2]),
    ];
    let sum: Scalar = angles.iter().sum();
    angles.map(|angle| angle / sum)
}

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
        weightings: &[[Scalar; 4]],
    ) -> HashMap<(usize, usize), UnnormalizedBondGradientVector> {
        let mut bonds: HashMap<(usize, usize), UnnormalizedBondGradientVector> = HashMap::new();
        connectivity
            .iter()
            .zip(elements.iter())
            .zip(weightings.iter())
            .for_each(|((nodes, element), weights)| {
                let gradient_vectors: &GradientVectors = element.gradient_vectors();
                nodes.iter().enumerate().for_each(|(index_a, &node_a)| {
                    let weight = element.volume() * weights[index_a];
                    nodes.iter().zip(gradient_vectors.iter()).for_each(
                        |(&node_b, gradient_vector_b)| {
                            if node_a != node_b {
                                let contribution = gradient_vector_b * weight;
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
        weighting: Weighting,
    ) -> Vec<Self> {
        let (elements, weightings): (Vec<Tetrahedron>, Vec<[Scalar; 4]>) = connectivity
            .iter()
            .map(|nodes| {
                let coordinates: ElementNodalReferenceCoordinates =
                    Self::element_coordinates(reference_coordinates, nodes);
                let weights = weighting.weights(&coordinates);
                (Tetrahedron::from(coordinates), weights)
            })
            .unzip();
        let mut volumes = vec![Quantity::<Volume>::new(0.0); reference_coordinates.len()];
        connectivity
            .iter()
            .zip(elements.iter())
            .zip(weightings.iter())
            .for_each(|((nodes, element), weights)| {
                nodes.iter().enumerate().for_each(|(index, &node)| {
                    volumes[node] += &(element.volume() * weights[index])
                })
            });
        let bonds = Self::accumulate_bonds(connectivity, &elements, &weightings);
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
