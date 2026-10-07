#[cfg(test)]
mod test;

use crate::{
    domain::{
        NodalReferenceCoordinates,
        mass::{LumpedMassElements, NodalLumpedMasses},
    },
    math::Tensor,
    vem::block::{Block, Densities, element::mass::LumpedMassVirtualElement},
};

impl<C, F, R> LumpedMassElements<3> for Block<C, F, R>
where
    F: LumpedMassVirtualElement,
    R: Densities,
{
    fn nodal_lumped_masses_into(
        &self,
        reference_coordinates: &NodalReferenceCoordinates<3>,
        nodal_lumped_masses: &mut NodalLumpedMasses,
    ) {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .enumerate()
            .for_each(|(element_index, (element, nodes))| {
                element
                    .nodal_lumped_masses(
                        self.density().at(element_index),
                        &Self::element_coordinates(reference_coordinates, nodes),
                    )
                    .iter()
                    .zip(nodes)
                    .for_each(|(&nodal_mass, &node)| nodal_lumped_masses[node] += nodal_mass)
            })
    }
}
