use crate::{
    constitutive::{ConstitutiveError, solid::elastic::Elastic},
    domain::{
        NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::element::solid::{SolidElement, elastic::ElasticElement},
    },
    fem::block::element::{
        ElementNodalReferenceCoordinates, FiniteElement, GradientVectors, linear::Tetrahedron,
    },
    geometry::mesh::PrimitiveConnectivity,
    math::{
        ContractSecondFourthWithFirst, Current, Quantity, Reference, Tensor, TensorRank1,
        TensorRank1List, TensorRank1Vec, TensorRank2,
    },
    mechanics::{DeformationGradient, DeformationGradientRate},
    units::{Force, ForcePerLength, ReciprocalLength, UnitMul, Volume},
};
use std::collections::HashMap;

/// A reference-configuration gradient vector, `\zeta_{ip}` in the source paper.
pub(crate) type BondGradientVector = TensorRank1<3, Reference, ReciprocalLength>;
/// The intermediate, unnormalized accumulation of a bond gradient vector across
/// the incident tetrahedra, before dividing through by the particle's volume.
type UnnormalizedBondGradientVector =
    TensorRank1<3, Reference, <ReciprocalLength as UnitMul<Volume>>::Output>;
/// A nodal force contribution, ordered the same as [`Node::gradient_vectors`].
type NodalForce = TensorRank1<3, Current, Force>;
/// A nodal stiffness block, ordered the same as [`Node::gradient_vectors`]
/// on both axes.
type NodalStiffness = TensorRank2<3, Current, Current, ForcePerLength>;

/// A particle: its reference volume and the bond gradient vectors — including
/// its own self term — of the neighbors spanning its bond neighborhood.
pub(crate) struct Node {
    volume: Quantity<Volume>,
    gradient_vectors: Vec<(usize, BondGradientVector)>,
}

impl Node {
    pub(crate) fn gradient_vectors(&self) -> &[(usize, BondGradientVector)] {
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
                                    .and_modify(|bond_gradient_vector| {
                                        *bond_gradient_vector += &contribution
                                    })
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
                    .map(|(_, bond_gradient_vector)| bond_gradient_vector)
                    .sum::<UnnormalizedBondGradientVector>();
                bonds.push((node, self_gradient_vector));
            });
        unnormalized
            .into_iter()
            .zip(volumes)
            .map(|(bonds, volume)| Node {
                volume,
                gradient_vectors: bonds
                    .into_iter()
                    .map(|(node, bond)| (node, bond / volume))
                    .collect(),
            })
            .collect()
    }
}

impl SolidElement for Node {
    type Coordinates = NodalCoordinates<3>;
    type Velocities = NodalVelocities<3>;
    type DeformationGradients = DeformationGradient;
    type DeformationGradientRates = DeformationGradientRate;
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> DeformationGradient {
        self.gradient_vectors
            .iter()
            .map(|(neighbor, bond_gradient_vector)| {
                DeformationGradient::from((&nodal_coordinates[*neighbor], bond_gradient_vector))
            })
            .sum()
    }
    fn deformation_gradient_rates(
        &self,
        _nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> DeformationGradientRate {
        self.gradient_vectors
            .iter()
            .map(|(neighbor, bond_gradient_vector)| {
                DeformationGradientRate::from((&nodal_velocities[*neighbor], bond_gradient_vector))
            })
            .sum()
    }
}

impl<C> ElasticElement<C> for Node
where
    C: Elastic,
{
    type Forces = Vec<NodalForce>;
    type Stiffnesses = Vec<Vec<NodalStiffness>>;
    type Error = ConstitutiveError;
    /// The forces this particle's stress contributes to each of its bonded
    /// neighbors (including itself), ordered the same as its bond list.
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Vec<NodalForce>, ConstitutiveError> {
        let first_piola_kirchhoff_stress = constitutive_model
            .first_piola_kirchhoff_stress(&self.deformation_gradients(nodal_coordinates))?;
        Ok(self
            .gradient_vectors
            .iter()
            .map(|(_, bond_gradient_vector)| {
                (&first_piola_kirchhoff_stress * bond_gradient_vector) * self.volume
            })
            .collect())
    }
    /// The stiffness blocks this particle's tangent contributes between each
    /// pair of its bonded neighbors (including itself), ordered the same as
    /// its bond list on both axes.
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Vec<Vec<NodalStiffness>>, ConstitutiveError> {
        let first_piola_kirchhoff_tangent_stiffness = constitutive_model
            .first_piola_kirchhoff_tangent_stiffness(
                &self.deformation_gradients(nodal_coordinates),
            )?;
        Ok(self
            .gradient_vectors
            .iter()
            .map(|(_, bond_gradient_vector_a)| {
                self.gradient_vectors
                    .iter()
                    .map(|(_, bond_gradient_vector_b)| {
                        first_piola_kirchhoff_tangent_stiffness.contract_second_fourth_with_first(
                            bond_gradient_vector_a,
                            bond_gradient_vector_b,
                        ) * self.volume
                    })
                    .collect()
            })
            .collect())
    }
}
