pub mod dielectric;

use crate::{
    fem::block::element::diffusion::{
        DiffusiveFiniteElement, ElementGradients, ElementNodalFields,
    },
    units::Charge,
};

pub type ElementNodalPotentials<const D: usize> = ElementNodalFields<Charge, D>;

pub trait ElectricalFiniteElement<const G: usize, const M: usize, const N: usize, const P: usize>
where
    Self: DiffusiveFiniteElement<Charge, G, M, N, P>,
{
    fn potential_gradients(
        &self,
        nodal_potentials: &ElementNodalPotentials<N>,
    ) -> ElementGradients<Charge, G> {
        self.gradients(nodal_potentials)
    }
}

impl<T, const G: usize, const M: usize, const N: usize, const P: usize>
    ElectricalFiniteElement<G, M, N, P> for T
where
    T: DiffusiveFiniteElement<Charge, G, M, N, P>,
{
}
