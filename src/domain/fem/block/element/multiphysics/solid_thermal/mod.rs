pub mod thermoelastic_thermal_conduction;

use crate::{
    fem::block::element::{
        solid::{ElementNodalStiffnessesSolid, SolidFiniteElement},
        thermal::{ThermalFiniteElement, conduction::ElementNodalStiffnessesThermal},
    },
    math::{Current, Quantity, TensorRank1, TensorRank1List2D, TensorTupleList},
    units::{Force, ForcePerTemperature, Power},
};

pub type ElementNodalForcesSolidThermal<const N: usize> =
    TensorTupleList<TensorRank1<3, Current, Force>, Quantity<Power>, N>;
pub type ElementNodalCouplingsSolidThermal<const N: usize> =
    TensorRank1List2D<3, Current, N, N, ForcePerTemperature>;
pub type ElementNodalHeatingsSolidThermal<const N: usize> =
    TensorRank1List2D<3, Current, N, N, Force>;
pub type ElementNodalStiffnessesSolidThermal<const N: usize> = (
    ElementNodalStiffnessesSolid<N>,
    ElementNodalCouplingsSolidThermal<N>,
    ElementNodalStiffnessesThermal<N>,
);

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
