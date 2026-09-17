//! Continuum bond methods

#[cfg(test)]
mod test;

use crate::{
    constitutive::{ConstitutiveError, solid::elastic::Elastic},
    domain::block::add_node_neighbors,
    fem::block::element::{
        ElementNodalReferenceCoordinates, FiniteElement, GradientVectors, linear::Tetrahedron,
    },
    geometry::mesh::PrimitiveConnectivity,
    math::{ContractSecondFourthWithFirst, Quantity, Reference, Tensor, TensorRank1},
    mechanics::{DeformationGradient, DeformationGradientRate},
    units::{ReciprocalLength, UnitMul, Volume},
};
use std::{
    collections::HashMap,
    fmt::{self, Debug, Formatter},
};

pub use crate::domain::{
    ElementModelError, FirstOrderRoot, NodalCoordinates, NodalReferenceCoordinates,
    NodalVelocities, ZerothOrderRoot,
    block::element::Elements,
    solid::{NodalForcesSolid, NodalStiffnessesSolid, SolidElements, elastic::ElasticElements},
};

/// A reference-configuration gradient vector, `\zeta_{ip}` in the source paper.
type BondGradientVector = TensorRank1<3, Reference, ReciprocalLength>;
/// The intermediate, unnormalized accumulation of a bond gradient vector across
/// the incident tetrahedra, before dividing through by the particle's volume.
type UnnormalizedBondGradientVector =
    TensorRank1<3, Reference, <ReciprocalLength as UnitMul<Volume>>::Output>;

pub struct Cbm<C> {
    constitutive_model: C,
    connectivity: PrimitiveConnectivity<3, 4>,
    node_volumes: Vec<Quantity<Volume>>,
    gradient_vectors: Vec<Vec<(usize, BondGradientVector)>>,
}

impl<C> Debug for Cbm<C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Cbm {{ {} particles }}", self.node_volumes.len())
    }
}

impl<C> Cbm<C> {
    fn element_coordinates<const D: usize, I, U>(
        coordinates: &crate::math::TensorRank1Vec<D, I, U>,
        nodes: &[usize; 4],
    ) -> crate::math::TensorRank1List<D, I, 4, U> {
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
}

impl<C>
    From<(
        C,
        PrimitiveConnectivity<3, 4>,
        &NodalReferenceCoordinates<3>,
    )> for Cbm<C>
{
    fn from(
        (constitutive_model, connectivity, reference_coordinates): (
            C,
            PrimitiveConnectivity<3, 4>,
            &NodalReferenceCoordinates<3>,
        ),
    ) -> Self {
        let elements: Vec<Tetrahedron> = connectivity
            .iter()
            .map(|nodes| -> ElementNodalReferenceCoordinates<4> {
                Self::element_coordinates(reference_coordinates, nodes)
            })
            .map(Tetrahedron::from)
            .collect();
        let mut node_volumes = vec![Quantity::<Volume>::new(0.0); reference_coordinates.len()];
        connectivity
            .iter()
            .zip(elements.iter())
            .for_each(|(nodes, element)| {
                let quarter_volume = element.volume() / 4.0;
                nodes
                    .iter()
                    .for_each(|&node| node_volumes[node] += &quarter_volume)
            });
        let bonds = Self::accumulate_bonds(&connectivity, &elements);
        let mut unnormalized: Vec<Vec<(usize, UnnormalizedBondGradientVector)>> =
            vec![Vec::new(); node_volumes.len()];
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
        let gradient_vectors: Vec<Vec<(usize, BondGradientVector)>> = unnormalized
            .into_iter()
            .zip(node_volumes.iter())
            .map(|(bonds, node_volume)| {
                bonds
                    .into_iter()
                    .map(|(node, bond)| (node, bond / *node_volume))
                    .collect()
            })
            .collect();
        Self {
            constitutive_model,
            connectivity,
            node_volumes,
            gradient_vectors,
        }
    }
}

impl<C> Elements for Cbm<C> {
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        add_node_neighbors(
            self.connectivity.iter().map(|nodes| nodes.as_slice()),
            neighbors,
        )
    }
}

impl<C> SolidElements for Cbm<C> {
    type DeformationGradients = DeformationGradient;
    type DeformationGradientRates = DeformationGradientRate;
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Vec<Self::DeformationGradients> {
        self.gradient_vectors
            .iter()
            .map(|bonds| {
                bonds
                    .iter()
                    .map(|(node, bond_gradient_vector)| {
                        DeformationGradient::from((&nodal_coordinates[*node], bond_gradient_vector))
                    })
                    .sum()
            })
            .collect()
    }
    fn deformation_gradient_rates(
        &self,
        _nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Vec<Self::DeformationGradientRates> {
        self.gradient_vectors
            .iter()
            .map(|bonds| {
                bonds
                    .iter()
                    .map(|(node, bond_gradient_vector)| {
                        DeformationGradientRate::from((
                            &nodal_velocities[*node],
                            bond_gradient_vector,
                        ))
                    })
                    .sum()
            })
            .collect()
    }
}

impl<C> ElasticElements<3> for Cbm<C>
where
    C: Elastic,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), ElementModelError> {
        SolidElements::deformation_gradients(self, nodal_coordinates)
            .iter()
            .zip(self.gradient_vectors.iter().zip(self.node_volumes.iter()))
            .try_for_each(|(deformation_gradient, (bonds, node_volume))| {
                let first_piola_kirchhoff_stress = self
                    .constitutive_model
                    .first_piola_kirchhoff_stress(deformation_gradient)?;
                bonds.iter().for_each(|(node, bond_gradient_vector)| {
                    nodal_forces[*node] +=
                        (&first_piola_kirchhoff_stress * bond_gradient_vector) * node_volume
                });
                Ok::<(), ConstitutiveError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<3>,
    ) -> Result<(), ElementModelError> {
        SolidElements::deformation_gradients(self, nodal_coordinates)
            .iter()
            .zip(self.gradient_vectors.iter().zip(self.node_volumes.iter()))
            .try_for_each(|(deformation_gradient, (bonds, node_volume))| {
                let first_piola_kirchhoff_tangent_stiffness = self
                    .constitutive_model
                    .first_piola_kirchhoff_tangent_stiffness(deformation_gradient)?;
                bonds.iter().for_each(|(node_a, bond_gradient_vector_a)| {
                    bonds.iter().for_each(|(node_b, bond_gradient_vector_b)| {
                        nodal_stiffnesses[*node_a][*node_b] +=
                            first_piola_kirchhoff_tangent_stiffness
                                .contract_second_fourth_with_first(
                                    bond_gradient_vector_a,
                                    bond_gradient_vector_b,
                                )
                                * node_volume
                    })
                });
                Ok::<(), ConstitutiveError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
