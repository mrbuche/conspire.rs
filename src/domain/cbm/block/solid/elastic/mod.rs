use super::super::{Block, node::solid::elastic::ElasticElement};
use crate::{
    constitutive::{ConstitutiveError, solid::elastic::Elastic},
    domain::{
        ElementModelError, NodalCoordinates,
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
};

pub use crate::domain::solid::elastic::ElasticElements;

impl<C> ElasticElements<3> for Block<C>
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
