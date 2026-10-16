use crate::{
    domain::{
        Model, SolverFor,
        diffusion::DiffusionElements,
        thermal::{NodalForcesThermal, NodalStiffnessesThermal},
    },
    math::{Quantity, optimize::NewtonRaphson},
    units::{PowerTemperature, Temperature},
};

pub trait ThermalConductionElements
where
    Self: DiffusionElements<Temperature>,
{
}

impl<B> ThermalConductionElements for B where B: DiffusionElements<Temperature> {}

impl<B, const D: usize> SolverFor<Model<B, D>, NodalForcesThermal, NodalStiffnessesThermal>
    for NewtonRaphson
where
    B: ThermalConductionElements,
{
    type Tangent = NodalStiffnessesThermal;
    const SPARSE: bool = true;
}

impl<B, const D: usize> SolverFor<Model<B, D>, Quantity<PowerTemperature>, NodalForcesThermal>
    for NewtonRaphson
where
    B: ThermalConductionElements,
{
    type Tangent = NodalStiffnessesThermal;
    const SPARSE: bool = true;
}
