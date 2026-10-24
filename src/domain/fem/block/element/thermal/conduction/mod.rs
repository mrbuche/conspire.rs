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

pub trait ThermalConductionFiniteElement<C, const G: usize, const M: usize, const N: usize>
where
    C: ThermalConduction,
    Self: DiffusionFiniteElement<C, G, M, N>,
{
}

impl<T, C, const G: usize, const M: usize, const N: usize>
    ThermalConductionFiniteElement<C, G, M, N> for T
where
    C: ThermalConduction,
    T: DiffusionFiniteElement<C, G, M, N>,
{
}
