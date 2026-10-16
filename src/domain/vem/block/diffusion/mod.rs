use crate::{
    constitutive::diffusion::{Diffusion, Diffusive},
    domain::{
        ElementModelError,
        diffusion::{DiffusionElements, NodalFields, NodalForces, NodalStiffnesses},
    },
    math::{Quantity, Tensor},
    vem::block::{
        Block,
        element::{VirtualElementError, diffusion::DiffusionVirtualElement},
    },
};

impl<C, F, R> Block<C, F, R> {
    pub(crate) fn element_fields<L>(
        nodes: &[usize],
        nodal_fields: &NodalFields<L>,
    ) -> NodalFields<L>
    where
        L: Diffusive,
    {
        nodes.iter().map(|&node| nodal_fields[node]).collect()
    }
}

impl<L, C, F, R> DiffusionElements<L> for Block<C, F, R>
where
    L: Diffusive,
    C: Diffusion<Balance = L>,
    F: DiffusionVirtualElement<C>,
{
    fn potential(
        &self,
        nodal_fields: &NodalFields<L>,
    ) -> Result<Quantity<L::Energy>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .map(|(element, nodes)| {
                element.potential(
                    self.constitutive_model(),
                    &Self::element_fields::<L>(nodes, nodal_fields),
                )
            })
            .sum::<Result<Quantity<L::Energy>, VirtualElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_forces_into(
        &self,
        nodal_fields: &NodalFields<L>,
        nodal_forces: &mut NodalForces<L>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &Self::element_fields::<L>(nodes, nodal_fields),
                    )?
                    .iter()
                    .zip(nodes)
                    .for_each(|(&nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), VirtualElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_fields: &NodalFields<L>,
        nodal_stiffnesses: &mut NodalStiffnesses<L>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_fields::<L>(nodes, nodal_fields),
                    )?
                    .iter()
                    .zip(nodes)
                    .for_each(|(row, &node_a)| {
                        row.iter()
                            .zip(nodes)
                            .for_each(|(&nodal_stiffness, &node_b)| {
                                nodal_stiffnesses[node_a][node_b] += nodal_stiffness
                            })
                    });
                Ok::<(), VirtualElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
