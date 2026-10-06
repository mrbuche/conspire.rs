#[cfg(test)]
mod test;

use crate::fem::{
    NodalReferenceCoordinates,
    block::{
        Block, Densities,
        element::mass::{LumpedMassFiniteElement, MassFiniteElement},
    },
    mass::{ConsistentMassElements, LumpedMassElements},
};

pub use crate::domain::mass::{NodalLumpedMasses, NodalMasses};

impl<C, F, R, const G: usize, const M: usize, const N: usize, const P: usize> ConsistentMassElements
    for Block<C, F, G, M, N, P, R>
where
    F: MassFiniteElement<G, M, N, P>,
    R: Densities<G>,
{
    fn nodal_masses_into(&self, nodal_masses: &mut NodalMasses) {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .enumerate()
            .for_each(|(element_index, (element, element_connectivity))| {
                element
                    .nodal_masses(&self.density().at(element_index))
                    .into_iter()
                    .zip(element_connectivity)
                    .for_each(|(object, &node_a)| {
                        object.into_iter().zip(element_connectivity).for_each(
                            |(nodal_mass, &node_b)| nodal_masses[node_a][node_b] += nodal_mass,
                        )
                    })
            })
    }
}

impl<C, F, R, const G: usize, const M: usize, const N: usize, const P: usize> LumpedMassElements<M>
    for Block<C, F, G, M, N, P, R>
where
    F: LumpedMassFiniteElement<G, M, N, P>,
    R: Densities<G>,
{
    fn nodal_lumped_masses_into(
        &self,
        _reference_coordinates: &NodalReferenceCoordinates<M>,
        nodal_lumped_masses: &mut NodalLumpedMasses,
    ) {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .enumerate()
            .for_each(|(element_index, (element, element_connectivity))| {
                element
                    .nodal_lumped_masses(&self.density().at(element_index))
                    .into_iter()
                    .zip(element_connectivity)
                    .for_each(|(nodal_mass, &node)| nodal_lumped_masses[node] += nodal_mass)
            })
    }
}
