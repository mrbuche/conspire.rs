#[cfg(test)]
pub mod test;

use super::temperature;
use crate::{
    constitutive::{
        ConstitutiveError, multiphysics::ThermoelasticThermalConduction,
        solid::thermoelastic::Thermoelastic, thermal::conduction::ThermalConduction,
    },
    fem::block::element::{
        Element, ElementNodalCoordinates, ElementNodalVelocities, FiniteElement,
        FiniteElementError,
        multiphysics::solid_thermal::{ElementNodalForcesSolidThermal, SolidThermalFiniteElement},
        solid::SolidElement,
        thermal::{ElementNodalTemperatures, ThermalFiniteElement},
    },
    math::{ContractWith, Tensor, TensorTuple},
};

pub trait ThermoelasticThermalConductionTransientFiniteElement<
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
> where
    C: ThermoelasticThermalConduction,
    Self: SolidThermalFiniteElement<G, M, N, P>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> Result<ElementNodalForcesSolidThermal<N>, FiniteElementError>;
}

impl<C, const G: usize, const N: usize, const O: usize>
    ThermoelasticThermalConductionTransientFiniteElement<C, G, 3, N, N> for Element<3, G, N, O>
where
    C: ThermoelasticThermalConduction,
    Self: SolidThermalFiniteElement<G, 3, N, N>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> Result<ElementNodalForcesSolidThermal<N>, FiniteElementError> {
        let deformation_gradients = self.deformation_gradients(nodal_coordinates);
        let deformation_gradient_rates =
            self.deformation_gradient_rates(nodal_coordinates, nodal_velocities);
        let temperature_gradients = self.temperature_gradients(nodal_temperatures);
        Self::shape_functions_at_integration_points()
            .iter()
            .zip(
                deformation_gradients.iter().zip(
                    deformation_gradient_rates
                        .iter()
                        .zip(temperature_gradients.iter()),
                ),
            )
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(
                |(
                    (
                        shape_functions,
                        (deformation_gradient, (deformation_gradient_rate, temperature_gradient)),
                    ),
                    (gradient_vectors, integration_weight),
                )| {
                    let temperature = temperature(shape_functions, nodal_temperatures);
                    let solid_model = constitutive_model.solid_constitutive_model();
                    let first_piola_kirchhoff_stress = solid_model
                        .first_piola_kirchhoff_stress(deformation_gradient, temperature)?;
                    let thermoelastic_heating = solid_model
                        .first_piola_kirchhoff_thermal_tangent(deformation_gradient, temperature)?
                        .contract_with(deformation_gradient_rate)
                        * integration_weight
                        * temperature;
                    let heat_flux = constitutive_model
                        .thermal_constitutive_model()
                        .heat_flux(temperature_gradient)?;
                    Ok(gradient_vectors
                        .iter()
                        .zip(shape_functions.iter())
                        .map(|(gradient_vector, shape_function)| {
                            TensorTuple(
                                (&first_piola_kirchhoff_stress * gradient_vector)
                                    * integration_weight,
                                -(&heat_flux * gradient_vector) * integration_weight
                                    - thermoelastic_heating * shape_function,
                            )
                        })
                        .collect::<ElementNodalForcesSolidThermal<N>>())
                },
            )
            .sum::<Result<_, ConstitutiveError>>()
            .map_err(|error| FiniteElementError::upstream(error, self))
    }
}
