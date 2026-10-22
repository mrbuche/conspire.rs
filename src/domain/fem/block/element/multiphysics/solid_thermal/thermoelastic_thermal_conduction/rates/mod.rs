#[cfg(test)]
pub mod test;

use super::steady::temperature;
use crate::{
    constitutive::{
        ConstitutiveError, multiphysics::ThermoelasticThermalConduction,
        solid::thermoelastic::Thermoelastic, thermal::conduction::ThermalConduction,
    },
    fem::block::element::{
        Element, ElementNodalCoordinates, ElementNodalVelocities, FiniteElement,
        FiniteElementError,
        multiphysics::solid_thermal::{
            ElementNodalForcesSolidThermal, ElementNodalHeatingStiffnessesSolidThermal,
            ElementNodalHeatingsSolidThermal, ElementNodalStiffnessesSolidThermal,
            SolidThermalFiniteElement,
            thermoelastic_thermal_conduction::steady::ThermoelasticThermalConductionFiniteElement,
        },
        solid::SolidElement,
        thermal::{ElementNodalTemperatures, ThermalFiniteElement},
    },
    math::{
        ContractSecondFourthWithFirst, ContractWith, Current, Rank2, Tensor, TensorRank1,
        TensorTuple,
    },
    units::PowerPerLength,
};

pub trait ThermoelasticThermalConductionRateFiniteElement<
    C,
    const G: usize,
    const M: usize,
    const N: usize,
> where
    C: ThermoelasticThermalConduction,
    Self: SolidThermalFiniteElement<G, M, N>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> Result<ElementNodalForcesSolidThermal<N>, FiniteElementError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> Result<
        (
            ElementNodalStiffnessesSolidThermal<N>,
            ElementNodalHeatingStiffnessesSolidThermal<N>,
            ElementNodalHeatingsSolidThermal<N>,
        ),
        FiniteElementError,
    >;
}

impl<C, const G: usize, const N: usize, const O: usize>
    ThermoelasticThermalConductionRateFiniteElement<C, G, 3, N> for Element<3, G, N, O>
where
    C: ThermoelasticThermalConduction,
    Self: SolidThermalFiniteElement<G, 3, N>,
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
                        .zip(&temperature_gradients),
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
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> Result<
        (
            ElementNodalStiffnessesSolidThermal<N>,
            ElementNodalHeatingStiffnessesSolidThermal<N>,
            ElementNodalHeatingsSolidThermal<N>,
        ),
        FiniteElementError,
    > {
        let (solid, coupling, mut thermal) =
            ThermoelasticThermalConductionFiniteElement::nodal_stiffnesses(
                self,
                constitutive_model,
                nodal_coordinates,
                nodal_temperatures,
            )?;
        let deformation_gradients = self.deformation_gradients(nodal_coordinates);
        let deformation_gradient_rates =
            self.deformation_gradient_rates(nodal_coordinates, nodal_velocities);
        let TensorTuple(heating_stiffnesses, heatings) =
            Self::shape_functions_at_integration_points()
                .iter()
                .zip(
                    deformation_gradients
                        .iter()
                        .zip(&deformation_gradient_rates),
                )
                .zip(
                    self.gradient_vectors()
                        .iter()
                        .zip(self.integration_weights()),
                )
                .map(
                    |(
                        (shape_functions, (deformation_gradient, deformation_gradient_rate)),
                        (gradient_vectors, integration_weight),
                    )| {
                        let temperature = temperature(shape_functions, nodal_temperatures);
                        let solid_model = constitutive_model.solid_constitutive_model();
                        let thermal_tangent = solid_model.first_piola_kirchhoff_thermal_tangent(
                            deformation_gradient,
                            temperature,
                        )?;
                        let thermal_tangent_stiffness = solid_model
                            .first_piola_kirchhoff_thermal_tangent_stiffness(
                                deformation_gradient,
                                temperature,
                            )?;
                        let mixed_tangent_stiffness =
                            solid_model.first_piola_kirchhoff_mixed_tangent_stiffness(
                                deformation_gradient,
                                temperature,
                            )? * temperature
                                * integration_weight;
                        let heating_rate = (thermal_tangent
                            .contract_with(deformation_gradient_rate)
                            + thermal_tangent_stiffness.contract_with(deformation_gradient_rate)
                                * temperature)
                            * integration_weight;
                        shape_functions.iter().zip(thermal.iter_mut()).for_each(
                            |(shape_function_a, thermal_a)| {
                                thermal_a.iter_mut().zip(shape_functions.iter()).for_each(
                                    |(thermal_ab, shape_function_b)| {
                                        *thermal_ab -=
                                            heating_rate * shape_function_a * shape_function_b
                                    },
                                )
                            },
                        );
                        let stiffness_column = |gradient_vector_b| {
                            gradient_vectors
                                .iter()
                                .zip(nodal_velocities)
                                .map(|(gradient_vector_c, velocity_c)| {
                                    mixed_tangent_stiffness
                                        .contract_second_fourth_with_first(
                                            gradient_vector_c,
                                            gradient_vector_b,
                                        )
                                        .transpose()
                                        * velocity_c
                                })
                                .sum::<TensorRank1<3, Current, PowerPerLength>>()
                        };
                        Ok(TensorTuple(
                            shape_functions
                                .iter()
                                .map(|shape_function_a| {
                                    gradient_vectors
                                        .iter()
                                        .map(|gradient_vector_b| {
                                            -stiffness_column(gradient_vector_b) * shape_function_a
                                        })
                                        .collect()
                                })
                                .collect::<ElementNodalHeatingStiffnessesSolidThermal<N>>(),
                            shape_functions
                                .iter()
                                .map(|shape_function_a| {
                                    gradient_vectors
                                        .iter()
                                        .map(|gradient_vector_b| {
                                            -((&thermal_tangent * gradient_vector_b)
                                                * integration_weight
                                                * temperature)
                                                * shape_function_a
                                        })
                                        .collect()
                                })
                                .collect::<ElementNodalHeatingsSolidThermal<N>>(),
                        ))
                    },
                )
                .sum::<Result<_, ConstitutiveError>>()
                .map_err(|error| FiniteElementError::upstream(error, self))?;
        Ok(((solid, coupling, thermal), heating_stiffnesses, heatings))
    }
}
