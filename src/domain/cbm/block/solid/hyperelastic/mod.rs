use super::super::{
    Block,
    node::solid::{elastic::ElasticElement, hyperelastic::HyperelasticElement},
};
use crate::{
    constitutive::{ConstitutiveError, solid::hyperelastic::Hyperelastic},
    domain::{ElementModelError, NodalCoordinates, solid::NodalStiffnessesSolidSymmetric},
    math::{HessianAccumulate, Quantity},
    units::Energy,
};

pub use crate::domain::solid::hyperelastic::HyperelasticElements;

impl<C> HyperelasticElements<3> for Block<C>
where
    C: Hyperelastic,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        self.nodes
            .iter()
            .map(|node| node.helmholtz_free_energy(&self.constitutive_model, nodal_coordinates))
            .sum::<Result<_, ConstitutiveError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_symmetric_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_stiffnesses: &mut NodalStiffnessesSolidSymmetric<3>,
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
                                if neighbor_a <= neighbor_b {
                                    nodal_stiffnesses.accumulate(neighbor_a, neighbor_b, block)
                                }
                            })
                    });
                Ok::<(), ConstitutiveError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
