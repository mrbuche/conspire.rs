#[cfg(test)]
mod test;

use crate::{
    cbm::node::Node,
    constitutive::{ConstitutiveError, solid::elastic::Elastic},
    domain::{
        ElementModelError, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::{add_node_neighbors, element::Elements},
        solid::{NodalForcesSolid, NodalStiffnessesSolid, SolidElements, elastic::ElasticElements},
    },
    geometry::mesh::PrimitiveConnectivity,
    math::ContractSecondFourthWithFirst,
    mechanics::{DeformationGradient, DeformationGradientRate},
};
use std::fmt::{self, Debug, Formatter};

pub struct Cbm<C> {
    constitutive_model: C,
    connectivity: PrimitiveConnectivity<3, 4>,
    nodes: Vec<Node>,
}

impl<C> Debug for Cbm<C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Cbm {{ {} particles }}", self.nodes.len())
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
        let nodes = Node::vec_from(&connectivity, reference_coordinates);
        Self {
            constitutive_model,
            connectivity,
            nodes,
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
        self.nodes
            .iter()
            .map(|node| {
                node.gradient_vectors()
                    .iter()
                    .map(|(neighbor, bond_gradient_vector)| {
                        DeformationGradient::from((
                            &nodal_coordinates[*neighbor],
                            bond_gradient_vector,
                        ))
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
        self.nodes
            .iter()
            .map(|node| {
                node.gradient_vectors()
                    .iter()
                    .map(|(neighbor, bond_gradient_vector)| {
                        DeformationGradientRate::from((
                            &nodal_velocities[*neighbor],
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
            .zip(self.nodes.iter())
            .try_for_each(|(deformation_gradient, node)| {
                let first_piola_kirchhoff_stress = self
                    .constitutive_model
                    .first_piola_kirchhoff_stress(deformation_gradient)?;
                node.gradient_vectors()
                    .iter()
                    .for_each(|(neighbor, bond_gradient_vector)| {
                        nodal_forces[*neighbor] +=
                            (&first_piola_kirchhoff_stress * bond_gradient_vector) * node.volume()
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
            .zip(self.nodes.iter())
            .try_for_each(|(deformation_gradient, node)| {
                let first_piola_kirchhoff_tangent_stiffness = self
                    .constitutive_model
                    .first_piola_kirchhoff_tangent_stiffness(deformation_gradient)?;
                node.gradient_vectors()
                    .iter()
                    .for_each(|(neighbor_a, bond_gradient_vector_a)| {
                        node.gradient_vectors().iter().for_each(
                            |(neighbor_b, bond_gradient_vector_b)| {
                                nodal_stiffnesses[*neighbor_a][*neighbor_b] +=
                                    first_piola_kirchhoff_tangent_stiffness
                                        .contract_second_fourth_with_first(
                                            bond_gradient_vector_a,
                                            bond_gradient_vector_b,
                                        )
                                        * node.volume()
                            },
                        )
                    });
                Ok::<(), ConstitutiveError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
