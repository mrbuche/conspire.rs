pub mod thermoelastic_thermal_conduction;

use crate::{
    fem::block::element::{solid::SolidFiniteElement, thermal::ThermalFiniteElement},
    math::{Current, Quantity, TensorRank1, TensorTupleList},
    units::{Entropy, Force},
};

pub type ElementNodalForcesSolidThermal<const N: usize> =
    TensorTupleList<TensorRank1<3, Current, Force>, Quantity<Entropy>, N>;

pub trait SolidThermalFiniteElement<const G: usize, const M: usize, const N: usize, const P: usize>
where
    Self: SolidFiniteElement<G, M, N, P> + ThermalFiniteElement<G, M, N, P>,
{
}

impl<T, const G: usize, const M: usize, const N: usize, const P: usize>
    SolidThermalFiniteElement<G, M, N, P> for T
where
    T: SolidFiniteElement<G, M, N, P> + ThermalFiniteElement<G, M, N, P>,
{
}
