use crate::domain::{
    ElementModelError, Model, ProvidesTangent,
    fem::{
        block::thermal::{NodalTemperatures, conduction::NodalStiffnessesThermal},
        thermal::conduction::ThermalConductionElements,
    },
};

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
