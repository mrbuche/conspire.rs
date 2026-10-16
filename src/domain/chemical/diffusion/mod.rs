use crate::{
    domain::{
        ElementModelError, Model, ProvidesTangent,
        diffusion::{DiffusionElements, NodalFields, NodalForces, NodalStiffnesses},
    },
    units::MolarFlowRate,
};

pub type NodalConcentrations = NodalFields<MolarFlowRate>;
pub type NodalMolarFlowRates = NodalForces<MolarFlowRate>;
pub type NodalVolumetricFlowRates = NodalStiffnesses<MolarFlowRate>;

pub trait ChemicalDiffusionElements
where
    Self: DiffusionElements<MolarFlowRate>,
{
}

impl<B> ChemicalDiffusionElements for B where B: DiffusionElements<MolarFlowRate> {}

impl<B, const D: usize> ProvidesTangent<NodalConcentrations, NodalVolumetricFlowRates>
    for Model<B, D>
where
    B: ChemicalDiffusionElements,
{
    fn provide_tangent(
        &self,
        nodal_concentrations: &NodalConcentrations,
    ) -> Result<NodalVolumetricFlowRates, ElementModelError> {
        self.nodal_stiffnesses(nodal_concentrations)
    }
}
