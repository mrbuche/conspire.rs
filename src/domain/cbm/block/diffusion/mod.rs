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

impl<U, C, R> DiffusionElements<U> for Block<C, R>
where
    U: Diffusive,
    C: Diffusion<Field = U>,
    Node: DiffusionNode<C>,
{
    fn potential(
        &self,
        nodal_fields: &NodalFields<U>,
    ) -> Result<Quantity<U::Energy>, ElementModelError> {
        self.nodes
            .iter()
            .map(|node| node.potential(&self.constitutive_model, nodal_fields))
            .sum::<Result<Quantity<U::Energy>, ConstitutiveError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_forces_into(
        &self,
        nodal_fields: &NodalFields<U>,
        nodal_forces: &mut NodalForces<U>,
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
        nodal_fields: &NodalFields<U>,
        nodal_stiffnesses: &mut NodalStiffnesses<U>,
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
