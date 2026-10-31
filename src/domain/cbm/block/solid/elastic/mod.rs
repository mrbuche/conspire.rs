use super::super::{Block, node::solid::elastic::ElasticElement};
use crate::{
    constitutive::{ConstitutiveError, solid::elastic::Elastic},
    domain::{
        ElementModelError, NodalCoordinates,
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
};

pub use crate::domain::solid::elastic::ElasticElements;

impl<C, R> ElasticElements<3> for Block<C, R>
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
                    .zip(node.neighbors())
                    .for_each(|(force, &neighbor)| nodal_forces[neighbor] += force);
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
}

/// TEMPORARY EXPERIMENT: a block with no parallel forces computes them serially.
impl<C, R> crate::domain::solid::elastic::ParallelElasticElements<3> for Block<C, R>
where
    C: Elastic,
{
    type Scratch = ();
    fn scratch(&self) {}
    fn nodal_forces_parallel_into(
        &self,
        _: &mut (),
        _: &crate::domain::solid::elastic::Workers,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), ElementModelError> {
        self.nodal_forces_into(nodal_coordinates, nodal_forces)
    }
}
