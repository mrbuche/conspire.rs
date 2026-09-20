use super::super::{Block, node::solid::viscoelastic::ViscoelasticElement};
use crate::{
    constitutive::{ConstitutiveError, solid::viscoelastic::Viscoelastic},
    domain::{
        ElementModelError, NodalCoordinates, NodalVelocities,
        solid::{NodalDampingsSolid, NodalForcesSolid},
    },
};

pub use crate::domain::solid::viscoelastic::ViscoelasticElements;

impl<C> ViscoelasticElements<3> for Block<C>
where
    C: Viscoelastic,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), ElementModelError> {
        self.nodes
            .iter()
            .try_for_each(|node| {
                node.nodal_forces(
                    &self.constitutive_model,
                    nodal_coordinates,
                    nodal_velocities,
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
        nodal_velocities: &NodalVelocities<3>,
        nodal_stiffnesses: &mut NodalDampingsSolid<3>,
    ) -> Result<(), ElementModelError> {
        self.nodes
            .iter()
            .try_for_each(|node| {
                node.nodal_stiffnesses(
                    &self.constitutive_model,
                    nodal_coordinates,
                    nodal_velocities,
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
}
