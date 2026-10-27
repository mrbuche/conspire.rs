#[cfg(test)]
pub mod test;

use crate::{
    constitutive::electrical::dielectric::Dielectric,
    fem::block::element::diffusion::{
        DiffusionFiniteElement, ElementNodalForces, ElementNodalStiffnesses,
    },
    units::Charge,
};

pub type ElementNodalForcesElectrical<const D: usize> = ElementNodalForces<Charge, D>;
pub type ElementNodalStiffnessesElectrical<const D: usize> = ElementNodalStiffnesses<Charge, D>;

pub trait DielectricFiniteElement<C, const G: usize, const M: usize, const N: usize>
where
    C: Dielectric,
    Self: DiffusionFiniteElement<C, G, M, N>,
{
}

impl<T, C, const G: usize, const M: usize, const N: usize> DielectricFiniteElement<C, G, M, N> for T
where
    C: Dielectric,
    T: DiffusionFiniteElement<C, G, M, N>,
{
}
