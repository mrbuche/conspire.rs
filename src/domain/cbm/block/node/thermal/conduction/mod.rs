use super::{super::Node, ThermalNode};
use crate::{
    constitutive::{ConstitutiveError, thermal::conduction::ThermalConduction},
    domain::thermal::NodalTemperatures,
    math::{ContractWith, Quantity},
    units::{Power, PowerPerTemperature, PowerTemperature},
};

pub trait ThermalConductionNode<C>
where
    C: ThermalConduction,
    Self: ThermalNode,
{
    fn potential(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Quantity<PowerTemperature>, ConstitutiveError>;
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Vec<Quantity<Power>>, ConstitutiveError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Vec<Vec<Quantity<PowerPerTemperature>>>, ConstitutiveError>;
}

impl<C> ThermalConductionNode<C> for Node
where
    C: ThermalConduction,
{
    fn potential(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Quantity<PowerTemperature>, ConstitutiveError> {
        Ok(
            constitutive_model.potential(&self.temperature_gradient(nodal_temperatures))?
                * self.volume,
        )
    }
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Vec<Quantity<Power>>, ConstitutiveError> {
        let heat_flux =
            constitutive_model.heat_flux(&self.temperature_gradient(nodal_temperatures))?;
        Ok(self
            .gradient_vectors()
            .iter()
            .map(|gradient_vector| -heat_flux.contract_with(gradient_vector) * self.volume)
            .collect())
    }
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Vec<Vec<Quantity<PowerPerTemperature>>>, ConstitutiveError> {
        let heat_flux_tangent =
            constitutive_model.heat_flux_tangent(&self.temperature_gradient(nodal_temperatures))?;
        Ok(self
            .gradient_vectors()
            .iter()
            .map(|gradient_vector_a| {
                self.gradient_vectors()
                    .iter()
                    .map(|gradient_vector_b| {
                        -gradient_vector_a.contract_with(&(&heat_flux_tangent * gradient_vector_b))
                            * self.volume
                    })
                    .collect()
            })
            .collect())
    }
}
