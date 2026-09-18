pub(crate) mod node;
#[cfg(test)]
mod test;

use crate::{
    constitutive::{ConstitutiveError, solid::elastic::Elastic},
    domain::{
        ElementModelError, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::{add_node_neighbors, element::Elements},
        solid::{NodalForcesSolid, NodalStiffnessesSolid, SolidElements, elastic::ElasticElements},
    },
    geometry::mesh::PrimitiveConnectivity,
    mechanics::{DeformationGradient, DeformationGradientRate},
};
use node::Node;
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
            .map(|node| node.deformation_gradient(nodal_coordinates))
            .collect()
    }
    fn deformation_gradient_rates(
        &self,
        _nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Vec<Self::DeformationGradientRates> {
        self.nodes
            .iter()
            .map(|node| node.deformation_gradient_rate(nodal_velocities))
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
        self.nodes
            .iter()
            .try_for_each(|node| {
                node.nodal_forces(&self.constitutive_model, nodal_coordinates)?
                    .into_iter()
                    .zip(node.gradient_vectors())
                    .for_each(|(force, (neighbor, _))| nodal_forces[*neighbor] += force);
                Ok::<(), ConstitutiveError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<3>,
    ) -> Result<(), ElementModelError> {
        self.nodes
            .iter()
            .try_for_each(|node| {
                node.nodal_stiffnesses(&self.constitutive_model, nodal_coordinates)?
                    .into_iter()
                    .zip(node.gradient_vectors())
                    .for_each(|(row, (neighbor_a, _))| {
                        row.into_iter().zip(node.gradient_vectors()).for_each(
                            |(block, (neighbor_b, _))| {
                                nodal_stiffnesses[*neighbor_a][*neighbor_b] += block
                            },
                        )
                    });
                Ok::<(), ConstitutiveError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
