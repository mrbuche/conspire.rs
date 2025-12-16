#[cfg(test)]
pub mod test;

use crate::{
    constitutive::{ConstitutiveError, thermal::conduction::ThermalConduction},
    fem::block::element::{
        Element, FiniteElementError,
        thermal::{ElementNodalTemperatures, ThermalFiniteElement},
    },
    math::{Scalar, Tensor, TensorRank0List, TensorRank0List2D},
    mechanics::{HeatFluxTangents, HeatFluxes},
};

pub type ElementNodalForcesThermal<const D: usize> = TensorRank0List<D>;
pub type ElementNodalStiffnessesThermal<const D: usize> = TensorRank0List2D<D>;

pub trait ThermalConductionFiniteElement<C, const G: usize, const N: usize>
where
    C: ThermalConduction,
    Self: ThermalFiniteElement<G, N>,
{
    fn potential(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> Result<Scalar, FiniteElementError>;
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

impl<C, const G: usize, const N: usize> ThermalConductionFiniteElement<C, G, N> for Element<G, N>
where
    C: ThermalConduction,
    Self: ThermalFiniteElement<G, N>,
{
    fn potential(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .temperature_gradients(nodal_temperatures)
            .iter()
            .zip(self.integration_weights())
            .map(|(temperature_gradient, integration_weight)| {
                Ok::<_, ConstitutiveError>(
                    constitutive_model.potential(temperature_gradient)? * integration_weight,
                )
            })
            .sum()
        {
            Ok(potential) => Ok(potential),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> Result<ElementNodalForcesThermal<N>, FiniteElementError> {
        match self
            .temperature_gradients(nodal_temperatures)
            .iter()
            .map(|temperature_gradient| constitutive_model.heat_flux(temperature_gradient))
            .collect::<Result<HeatFluxes<G>, _>>()
        {
            Ok(heat_fluxes) => Ok(heat_fluxes
                .iter()
                .zip(
                    self.gradient_vectors()
                        .iter()
                        .zip(self.integration_weights()),
                )
                .map(|(heat_flux, (gradient_vectors, integration_weight))| {
                    gradient_vectors
                        .iter()
                        .map(|gradient_vector| -(heat_flux * gradient_vector) * integration_weight)
                        .collect()
                })
                .sum()),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> Result<ElementNodalStiffnessesThermal<N>, FiniteElementError> {
        match self
            .temperature_gradients(nodal_temperatures)
            .iter()
            .map(|temperature_gradient| constitutive_model.heat_flux_tangent(temperature_gradient))
            .collect::<Result<HeatFluxTangents<G>, _>>()
        {
            Ok(heat_flux_tangents) => Ok(heat_flux_tangents
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
                                        -(gradient_vector_a
                                            * (heat_flux_tangent * gradient_vector_b))
                                            * integration_weight
                                    })
                                    .collect()
                            })
                            .collect()
                    },
                )
                .sum()),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
