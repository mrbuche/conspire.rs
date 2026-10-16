use crate::{
    domain::{
        ElementModelError, Model, ProvidesTangent,
        diffusion::{DiffusionElements, NodalForces, NodalStiffnesses},
        electrical::NodalPotentials,
    },
    units::Current,
};

pub type NodalCurrents = NodalForces<Current>;
pub type NodalConductances = NodalStiffnesses<Current>;

pub trait ElectricalConductionElements
where
    Self: DiffusionElements<Current>,
{
}

impl<B> ElectricalConductionElements for B where B: DiffusionElements<Current> {}

impl<B, const D: usize> ProvidesTangent<NodalPotentials, NodalConductances> for Model<B, D>
where
    B: ElectricalConductionElements,
{
    fn provide_tangent(
        &self,
        nodal_potentials: &NodalPotentials,
    ) -> Result<NodalConductances, ElementModelError> {
        self.nodal_stiffnesses(nodal_potentials)
    }
}
