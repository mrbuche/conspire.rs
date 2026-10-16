use crate::{
    domain::{
        ElementModelError, Model, ProvidesTangent,
        diffusion::DiffusionElements,
        electrical::{NodalPotentials, NodalStiffnessesElectrical},
    },
    units::Charge,
};

pub trait DielectricElements
where
    Self: DiffusionElements<Charge>,
{
}

impl<B> DielectricElements for B where B: DiffusionElements<Charge> {}

impl<B, const D: usize> ProvidesTangent<NodalPotentials, NodalStiffnessesElectrical> for Model<B, D>
where
    B: DielectricElements,
{
    fn provide_tangent(
        &self,
        nodal_potentials: &NodalPotentials,
    ) -> Result<NodalStiffnessesElectrical, ElementModelError> {
        self.nodal_stiffnesses(nodal_potentials)
    }
}
