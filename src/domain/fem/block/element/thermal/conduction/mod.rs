#[cfg(test)]
pub mod test;

use crate::{
    constitutive::thermal::conduction::ThermalConduction,
    fem::block::element::diffusion::{
        DiffusionFiniteElement, ElementNodalForces, ElementNodalStiffnesses,
    },
    units::Power,
};

pub type ElementNodalForcesThermal<const D: usize> = ElementNodalForces<Power, D>;
pub type ElementNodalStiffnessesThermal<const D: usize> = ElementNodalStiffnesses<Power, D>;

pub trait ThermalConductionFiniteElement<
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
> where
    C: ThermalConduction,
    Self: DiffusionFiniteElement<C, G, M, N, P>,
{
}

impl<T, C, const G: usize, const M: usize, const N: usize, const P: usize>
    ThermalConductionFiniteElement<C, G, M, N, P> for T
where
    C: ThermalConduction,
    T: DiffusionFiniteElement<C, G, M, N, P>,
{
}
