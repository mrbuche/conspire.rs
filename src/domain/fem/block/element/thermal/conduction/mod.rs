#[cfg(test)]
pub mod test;

use crate::{
    constitutive::{ConstitutiveError, thermal::conduction::ThermalConduction},
    fem::block::element::{
        Element, FiniteElement, FiniteElementError,
        thermal::{ElementNodalTemperatures, ThermalFiniteElement},
    },
    math::{ContractWith, Quantity, Tensor, TensorList},
    mechanics::{HeatFluxTangents, HeatFluxes},
    units::{Power, PowerPerTemperature, PowerTemperature},
};

pub type ElementNodalForcesThermal<const D: usize> = TensorList<Quantity<Power>, D>;
pub type ElementNodalStiffnessesThermal<const D: usize> =
    TensorList<ElementNodalForcesThermalPerTemperature<D>, D>;
pub type ElementNodalForcesThermalPerTemperature<const D: usize> =
    TensorList<Quantity<PowerPerTemperature>, D>;

pub trait ThermalConductionFiniteElement<
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
> where
    C: ThermalConduction,
    Self: ThermalFiniteElement<G, M, N, P>,
{
    fn potential(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> Result<Quantity<PowerTemperature>, FiniteElementError>;
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> Result<ElementNodalForcesThermal<N>, FiniteElementError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> Result<ElementNodalStiffnessesThermal<N>, FiniteElementError>;
}

impl<C, const G: usize, const M: usize, const N: usize, const O: usize, const P: usize>
    ThermalConductionFiniteElement<C, G, M, N, P> for Element<3, G, N, O>
where
    C: ThermalConduction,
    Self: ThermalFiniteElement<G, M, N, P>,
{
    fn potential(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> Result<Quantity<PowerTemperature>, FiniteElementError> {
        self.temperature_gradients(nodal_temperatures)
            .iter()
            .zip(self.integration_weights())
            .map(|(temperature_gradient, integration_weight)| {
                Ok::<_, ConstitutiveError>(
                    constitutive_model.potential(temperature_gradient)? * integration_weight,
                )
            })
            .sum::<Result<_, ConstitutiveError>>()
            .map_err(|error| FiniteElementError::upstream(error, self))
    }
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> Result<ElementNodalForcesThermal<N>, FiniteElementError> {
        let heat_fluxes = self
            .temperature_gradients(nodal_temperatures)
            .iter()
            .map(|temperature_gradient| constitutive_model.heat_flux(temperature_gradient))
            .collect::<Result<HeatFluxes<G>, _>>()
            .map_err(|error| FiniteElementError::upstream(error, self))?;
        Ok(heat_fluxes
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(|(heat_flux, (gradient_vectors, integration_weight))| {
                gradient_vectors
                    .iter()
                    .map(|gradient_vector| {
                        -heat_flux.contract_with(gradient_vector) * integration_weight
                    })
                    .collect()
            })
            .sum())
    }
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> Result<ElementNodalStiffnessesThermal<N>, FiniteElementError> {
        let heat_flux_tangents = self
            .temperature_gradients(nodal_temperatures)
            .iter()
            .map(|temperature_gradient| constitutive_model.heat_flux_tangent(temperature_gradient))
            .collect::<Result<HeatFluxTangents<G>, _>>()
            .map_err(|error| FiniteElementError::upstream(error, self))?;
        Ok(heat_flux_tangents
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(
                |(heat_flux_tangent, (gradient_vectors, integration_weight))| {
                    gradient_vectors
                        .iter()
                        .map(|gradient_vector_a| {
                            gradient_vectors
                                .iter()
                                .map(|gradient_vector_b| {
                                    -gradient_vector_a
                                        .contract_with(&(heat_flux_tangent * gradient_vector_b))
                                        * integration_weight
                                })
                                .collect()
                        })
                        .collect()
                },
            )
            .sum())
    }
}
