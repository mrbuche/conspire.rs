pub mod dielectric;

use crate::{
    fem::block::element::diffusion::{
        DiffusiveFiniteElement, ElementGradients, ElementNodalFields,
    },
    units::ElectricPotential,
};

pub type ElementNodalPotentials<const D: usize> = ElementNodalFields<ElectricPotential, D>;

pub trait ElectricalFiniteElement<const G: usize, const M: usize, const N: usize, const P: usize>
where
    Self: DiffusiveFiniteElement<ElectricPotential, G, M, N, P>,
{
    fn potential_gradients(
        &self,
        nodal_potentials: &ElementNodalPotentials<N>,
    ) -> ElementGradients<ElectricPotential, G> {
        self.gradients(nodal_potentials)
    }
}

impl<T, const G: usize, const M: usize, const N: usize, const P: usize>
    ElectricalFiniteElement<G, M, N, P> for T
where
    T: DiffusiveFiniteElement<ElectricPotential, G, M, N, P>,
{
}
