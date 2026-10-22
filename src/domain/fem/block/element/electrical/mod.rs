pub mod dielectric;

use crate::{
    fem::block::element::diffusion::{
        DiffusiveFiniteElement, ElementGradients, ElementNodalFields,
    },
    units::Charge,
};

pub type ElementNodalPotentials<const D: usize> = ElementNodalFields<Charge, D>;

pub trait ElectricalFiniteElement<const G: usize, const M: usize, const N: usize>
where
    Self: DiffusiveFiniteElement<Charge, G, M, N>,
{
    fn potential_gradients(
        &self,
        nodal_potentials: &ElementNodalPotentials<N>,
    ) -> ElementGradients<Charge, G> {
        self.gradients(nodal_potentials)
    }
}

impl<T, const G: usize, const M: usize, const N: usize> ElectricalFiniteElement<G, M, N> for T where
    T: DiffusiveFiniteElement<Charge, G, M, N>
{
}
