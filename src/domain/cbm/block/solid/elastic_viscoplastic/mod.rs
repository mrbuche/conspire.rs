use super::super::{Block, node::solid::elastic_viscoplastic::ElasticViscoplasticElement};
use crate::{
    constitutive::{ConstitutiveError, solid::elastic_viscoplastic::ElasticViscoplastic},
    domain::{
        ElementModelError, NodalCoordinates,
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    math::{Differentiable, Tensor},
};

pub use crate::domain::block::solid::viscoplastic::{
    ViscoplasticEvolution, ViscoplasticStateVariables,
};
pub use crate::domain::solid::elastic_viscoplastic::{
    ElasticViscoplasticBCs, ElasticViscoplasticElements,
};

impl<C, Y> ElasticViscoplasticElements<ViscoplasticStateVariables<1, Y>, 3> for Block<C>
where
    C: ElasticViscoplastic<Y>,
    Y: Differentiable + Tensor,
{
    fn initial_state(&self) -> ViscoplasticStateVariables<1, Y> {
        self.nodes
            .iter()
            .map(|_| [self.constitutive_model.initial_state()].into())
            .collect()
    }
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &ViscoplasticStateVariables<1, Y>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), ElementModelError> {
        self.nodes
            .iter()
            .zip(state_variables)
            .try_for_each(|(node, state_variables_node)| {
                node.nodal_forces(
                    &self.constitutive_model,
                    nodal_coordinates,
                    state_variables_node,
                )?
                .into_iter()
                .zip(node.neighbors())
                .for_each(|(force, &neighbor)| nodal_forces[neighbor] += force);
                Ok::<(), ConstitutiveError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &ViscoplasticStateVariables<1, Y>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<3>,
    ) -> Result<(), ElementModelError> {
        self.nodes
            .iter()
            .zip(state_variables)
            .try_for_each(|(node, state_variables_node)| {
                node.nodal_stiffnesses(
                    &self.constitutive_model,
                    nodal_coordinates,
                    state_variables_node,
                )?
                .into_iter()
                .zip(node.neighbors())
                .for_each(|(row, &neighbor_a)| {
                    row.into_iter()
                        .zip(node.neighbors())
                        .for_each(|(block, &neighbor_b)| {
                            nodal_stiffnesses[neighbor_a][neighbor_b] += block
                        })
                });
                Ok::<(), ConstitutiveError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &ViscoplasticStateVariables<1, Y>,
    ) -> Result<ViscoplasticEvolution<1, Y>, ElementModelError> {
        self.nodes
            .iter()
            .zip(state_variables)
            .map(|(node, state_variables_node)| {
                node.state_variables_evolution(
                    &self.constitutive_model,
                    nodal_coordinates,
                    state_variables_node,
                )
            })
            .collect::<Result<_, ConstitutiveError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
