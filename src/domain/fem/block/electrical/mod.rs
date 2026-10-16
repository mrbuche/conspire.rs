pub mod dielectric;

use crate::{
    constitutive::electrical::Electrical,
    fem::block::{
        Block,
        element::electrical::{ElectricalFiniteElement, ElementNodalPotentials},
    },
    mechanics::ElectricFields,
};

pub use crate::domain::electrical::NodalPotentials;

pub trait ElectricalElements<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
where
    C: Electrical,
    F: ElectricalFiniteElement<G, M, N, P>,
{
    fn nodal_potentials_element(
        &self,
        element_connectivity: &[usize; N],
        nodal_potentials: &NodalPotentials,
    ) -> ElementNodalPotentials<N>;
    fn potential_gradients(&self, nodal_potentials: &NodalPotentials) -> Vec<ElectricFields<G>>;
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, R, H>
    ElectricalElements<C, F, G, M, N, P> for Block<C, F, G, M, N, P, R, H>
where
    C: Electrical,
    F: ElectricalFiniteElement<G, M, N, P>,
{
    fn nodal_potentials_element(
        &self,
        element_connectivity: &[usize; N],
        nodal_potentials: &NodalPotentials,
    ) -> ElementNodalPotentials<N> {
        element_connectivity
            .iter()
            .map(|&node| nodal_potentials[node])
            .collect()
    }
    fn potential_gradients(&self, nodal_potentials: &NodalPotentials) -> Vec<ElectricFields<G>> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, element_connectivity)| {
                element.potential_gradients(
                    &self.nodal_potentials_element(element_connectivity, nodal_potentials),
                )
            })
            .collect()
    }
}
