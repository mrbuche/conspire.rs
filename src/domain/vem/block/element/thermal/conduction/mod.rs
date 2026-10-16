#[cfg(test)]
mod test;

use crate::{
    constitutive::thermal::conduction::ThermalConduction,
    units::Temperature,
    vem::block::element::diffusion::{
        DiffusionVirtualElement, ElementNodalForces, ElementNodalStiffnesses,
    },
};

pub type ElementNodalForcesThermal = ElementNodalForces<Temperature>;
pub type ElementNodalStiffnessesThermal = ElementNodalStiffnesses<Temperature>;

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
