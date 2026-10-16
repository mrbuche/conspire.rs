use super::{
    Block,
    node::{Node, diffusion::DiffusionNode},
};
use crate::{
    constitutive::{
        ConstitutiveError,
        diffusion::{Diffusion, Diffusive},
    },
    domain::{
        ElementModelError,
        diffusion::{DiffusionElements, NodalFields, NodalForces, NodalStiffnesses},
    },
    math::Quantity,
};

impl<L, C, R> DiffusionElements<L> for Block<C, R>
where
    L: Diffusive,
    C: Diffusion<Balance = L>,
    Node: DiffusionNode<C>,
{
    fn potential(
        &self,
        nodal_fields: &NodalFields<L>,
    ) -> Result<Quantity<L::Energy>, ElementModelError> {
        self.nodes
            .iter()
            .map(|node| node.potential(&self.constitutive_model, nodal_fields))
            .sum::<Result<Quantity<L::Energy>, ConstitutiveError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_forces_into(
        &self,
        nodal_fields: &NodalFields<L>,
        nodal_forces: &mut NodalForces<L>,
    ) -> Result<(), ElementModelError> {
        self.nodes
            .iter()
            .try_for_each(|node| {
                node.nodal_forces(&self.constitutive_model, nodal_fields)?
                    .into_iter()
                    .zip(node.neighbors())
                    .for_each(|(force, &neighbor)| nodal_forces[neighbor] += force);
                Ok::<(), ConstitutiveError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_fields: &NodalFields<L>,
        nodal_stiffnesses: &mut NodalStiffnesses<L>,
    ) -> Result<(), ElementModelError> {
        self.nodes
            .iter()
            .try_for_each(|node| {
                node.nodal_stiffnesses(&self.constitutive_model, nodal_fields)?
                    .into_iter()
                    .zip(node.neighbors())
                    .for_each(|(row, &neighbor_a)| {
                        row.into_iter().zip(node.neighbors()).for_each(
                            |(stiffness, &neighbor_b)| {
                                nodal_stiffnesses[neighbor_a][neighbor_b] += stiffness
                            },
                        )
                    });
                Ok::<(), ConstitutiveError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
