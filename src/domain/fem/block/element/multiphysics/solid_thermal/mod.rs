pub mod thermoelastic_thermal_conduction;

use crate::{
    fem::block::element::{
        solid::{ElementNodalStiffnessesSolid, SolidFiniteElement},
        thermal::{ThermalFiniteElement, conduction::ElementNodalStiffnessesThermal},
    },
    math::{Current, Quantity, TensorRank1, TensorRank1List2D, TensorTupleList},
    units::{Force, ForcePerTemperature, Power, PowerPerLength},
};

pub type ElementNodalForcesSolidThermal<const N: usize> =
    TensorTupleList<TensorRank1<3, Current, Force>, Quantity<Power>, N>;
pub type ElementNodalCouplingsSolidThermal<const N: usize> =
    TensorRank1List2D<3, Current, N, N, ForcePerTemperature>;
pub type ElementNodalHeatingsSolidThermal<const N: usize> =
    TensorRank1List2D<3, Current, N, N, Force>;
pub type ElementNodalHeatingStiffnessesSolidThermal<const N: usize> =
    TensorRank1List2D<3, Current, N, N, PowerPerLength>;
pub type ElementNodalStiffnessesSolidThermal<const N: usize> = (
    ElementNodalStiffnessesSolid<N>,
    ElementNodalCouplingsSolidThermal<N>,
    ElementNodalStiffnessesThermal<N>,
);

pub trait SolidThermalFiniteElement<const G: usize, const M: usize, const N: usize>
where
    Self: SolidFiniteElement<G, M, N> + ThermalFiniteElement<G, M, N>,
{
}

impl<T, const G: usize, const M: usize, const N: usize> SolidThermalFiniteElement<G, M, N> for T where
    T: SolidFiniteElement<G, M, N> + ThermalFiniteElement<G, M, N>
{
}
