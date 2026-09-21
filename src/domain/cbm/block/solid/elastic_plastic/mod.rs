use super::super::{Block, node::solid::elastic_plastic::ElasticPlasticElement};
use crate::{
    constitutive::{ConstitutiveError, solid::elastic_plastic::ElasticPlastic},
    domain::{
        ElementModelError, NodalCoordinates,
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
};

pub use crate::domain::block::solid::plastic::PlasticStateVariablesField;
pub use crate::domain::solid::elastic_plastic::{ElasticPlasticElements, ElasticPlasticRoot};

impl<C> ElasticPlasticElements<PlasticStateVariablesField<1>, 3> for Block<C>
where
    C: ElasticPlastic,
{
    fn initial_state(&self) -> PlasticStateVariablesField<1> {
        self.nodes
            .iter()
            .map(|_| [self.constitutive_model.initial_state()].into())
            .collect()
    }
    fn nodal_forces_and_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &PlasticStateVariablesField<1>,
        nodal_forces: &mut NodalForcesSolid<3>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<3>,
    ) -> Result<(), ElementModelError> {
        self.nodes
            .iter()
            .zip(state_variables)
            .try_for_each(|(node, state_variables_node)| {
                let (forces, stiffnesses) = node.nodal_forces_and_stiffnesses(
                    &self.constitutive_model,
                    nodal_coordinates,
                    state_variables_node,
                )?;
                forces
                    .into_iter()
                    .zip(node.neighbors())
                    .for_each(|(force, &neighbor)| nodal_forces[neighbor] += force);
                stiffnesses
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
    fn updated_state(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &PlasticStateVariablesField<1>,
    ) -> Result<PlasticStateVariablesField<1>, ElementModelError> {
        self.nodes
            .iter()
            .zip(state_variables)
            .map(|(node, state_variables_node)| {
                node.updated_state(
                    &self.constitutive_model,
                    nodal_coordinates,
                    state_variables_node,
                )
            })
            .collect::<Result<_, ConstitutiveError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
