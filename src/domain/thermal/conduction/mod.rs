use crate::{
    domain::{
        ElementModelError, Model, ProvidesTangent,
        diffusion::DiffusionElements,
        thermal::{NodalStiffnessesThermal, NodalTemperatures},
    },
    units::Power,
};

pub trait ThermalConductionElements
where
    Self: DiffusionElements<Power>,
{
}

impl<B> ThermalConductionElements for B where B: DiffusionElements<Power> {}

impl<B, const D: usize> ProvidesTangent<NodalTemperatures, NodalStiffnessesThermal> for Model<B, D>
where
    B: ThermalConductionElements,
{
    fn provide_tangent(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalStiffnessesThermal, ElementModelError> {
        self.nodal_stiffnesses(nodal_temperatures)
    }
}
