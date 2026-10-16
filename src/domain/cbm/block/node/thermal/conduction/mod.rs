use super::super::diffusion::DiffusionNode;
use crate::constitutive::thermal::conduction::ThermalConduction;

pub trait ThermalConductionNode<C>
where
    C: ThermalConduction,
    Self: DiffusionNode<C>,
{
}

impl<T, C> ThermalConductionNode<C> for T
where
    C: ThermalConduction,
    T: DiffusionNode<C>,
{
}
