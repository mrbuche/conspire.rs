use crate::{
    domain::{
        Model, SolverFor,
        diffusion::DiffusionElements,
        electrical::{NodalForcesElectrical, NodalStiffnessesElectrical},
    },
    math::{Quantity, optimize::NewtonRaphson},
    units::{ElectricPotential, Energy},
};

pub trait DielectricElements
where
    Self: DiffusionElements<ElectricPotential>,
{
}

impl<B> DielectricElements for B where B: DiffusionElements<ElectricPotential> {}

impl<B, const D: usize> SolverFor<Model<B, D>, NodalForcesElectrical, NodalStiffnessesElectrical>
    for NewtonRaphson
where
    B: DielectricElements,
{
    type Tangent = NodalStiffnessesElectrical;
    const SPARSE: bool = true;
}

impl<B, const D: usize> SolverFor<Model<B, D>, Quantity<Energy>, NodalForcesElectrical>
    for NewtonRaphson
where
    B: DielectricElements,
{
    type Tangent = NodalStiffnessesElectrical;
    const SPARSE: bool = true;
}
