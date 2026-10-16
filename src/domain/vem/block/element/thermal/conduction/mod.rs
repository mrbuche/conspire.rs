#[cfg(test)]
mod test;

use crate::{
    constitutive::thermal::conduction::ThermalConduction,
    units::Power,
    vem::block::element::diffusion::{
        DiffusionVirtualElement, ElementNodalForces, ElementNodalStiffnesses,
    },
};

pub type ElementNodalForcesThermal = ElementNodalForces<Power>;
pub type ElementNodalStiffnessesThermal = ElementNodalStiffnesses<Power>;

pub trait ThermalConductionVirtualElement<C>
where
    C: ThermalConduction,
    Self: DiffusionVirtualElement<C>,
{
}

impl<T, C> ThermalConductionVirtualElement<C> for T
where
    C: ThermalConduction,
    T: DiffusionVirtualElement<C>,
{
}
