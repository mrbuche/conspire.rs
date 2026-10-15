#[cfg(test)]
pub mod test;

use crate::{
    constitutive::{
        ConstitutiveError, multiphysics::ThermoelasticThermalConduction,
        solid::thermoelastic::Thermoelastic, thermal::conduction::ThermalConduction,
    },
    fem::block::element::{
        Element, ElementNodalCoordinates, FiniteElement, FiniteElementError, ShapeFunctions,
        multiphysics::solid_thermal::{
            ElementNodalForcesSolidThermal, ElementNodalStiffnessesSolidThermal,
            SolidThermalFiniteElement,
        },
        solid::SolidElement,
        thermal::{ElementNodalTemperatures, ThermalFiniteElement},
    },
    math::{ContractSecondFourthWithFirst, Quantity, Tensor, TensorTuple},
    units::Temperature,
};

pub trait ThermoelasticThermalConductionFiniteElement<
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
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> Result<ElementNodalForcesSolidThermal<N>, FiniteElementError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> Result<ElementNodalStiffnessesSolidThermal<N>, FiniteElementError>;
}

pub(super) fn temperature<const N: usize>(
    shape_functions: &ShapeFunctions<N>,
    nodal_temperatures: &ElementNodalTemperatures<N>,
) -> Quantity<Temperature> {
    nodal_temperatures
        .iter()
        .zip(shape_functions.iter())
        .map(|(nodal_temperature, shape_function)| nodal_temperature * shape_function)
        .sum()
}

impl<C, const G: usize, const N: usize, const O: usize>
    ThermoelasticThermalConductionFiniteElement<C, G, 3, N, N> for Element<3, G, N, O>
where
    C: ThermoelasticThermalConduction,
    Self: SolidThermalFiniteElement<G, 3, N, N>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> Result<ElementNodalForcesSolidThermal<N>, FiniteElementError> {
        let deformation_gradients = self.deformation_gradients(nodal_coordinates);
        let temperature_gradients = self.temperature_gradients(nodal_temperatures);
        Self::shape_functions_at_integration_points()
            .iter()
            .zip(
                deformation_gradients
                    .iter()
                    .zip(temperature_gradients.iter()),
            )
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(
                |(
                    (shape_functions, (deformation_gradient, temperature_gradient)),
                    (gradient_vectors, integration_weight),
                )| {
                    let first_piola_kirchhoff_stress = constitutive_model
                        .solid_constitutive_model()
                        .first_piola_kirchhoff_stress(
                            deformation_gradient,
                            temperature(shape_functions, nodal_temperatures),
                        )?;
                    let heat_flux = constitutive_model
                        .thermal_constitutive_model()
                        .heat_flux(temperature_gradient)?;
                    Ok(gradient_vectors
                        .iter()
                        .map(|gradient_vector| {
                            TensorTuple(
                                (&first_piola_kirchhoff_stress * gradient_vector)
                                    * integration_weight,
                                -(&heat_flux * gradient_vector) * integration_weight,
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
        nodal_temperatures: &ElementNodalTemperatures<N>,
    ) -> Result<ElementNodalStiffnessesSolidThermal<N>, FiniteElementError> {
        let deformation_gradients = self.deformation_gradients(nodal_coordinates);
        let temperature_gradients = self.temperature_gradients(nodal_temperatures);
        let TensorTuple(TensorTuple(solid, coupling), thermal) =
            Self::shape_functions_at_integration_points()
                .iter()
                .zip(
                    deformation_gradients
                        .iter()
                        .zip(temperature_gradients.iter()),
                )
                .zip(
                    self.gradient_vectors()
                        .iter()
                        .zip(self.integration_weights()),
                )
                .map(
                    |(
                        (shape_functions, (deformation_gradient, temperature_gradient)),
                        (gradient_vectors, integration_weight),
                    )| {
                        let temperature = temperature(shape_functions, nodal_temperatures);
                        let solid_model = constitutive_model.solid_constitutive_model();
                        let first_piola_kirchhoff_tangent_stiffness = solid_model
                            .first_piola_kirchhoff_tangent_stiffness(
                                deformation_gradient,
                                temperature,
                            )?;
                        let first_piola_kirchhoff_thermal_tangent = solid_model
                            .first_piola_kirchhoff_thermal_tangent(
                                deformation_gradient,
                                temperature,
                            )?;
                        let heat_flux_tangent = constitutive_model
                            .thermal_constitutive_model()
                            .heat_flux_tangent(temperature_gradient)?;
                        Ok(TensorTuple(
                            TensorTuple(
                                gradient_vectors
                                    .iter()
                                    .map(|gradient_vector_a| {
                                        gradient_vectors
                                            .iter()
                                            .map(|gradient_vector_b| {
                                                first_piola_kirchhoff_tangent_stiffness
                                                    .contract_second_fourth_with_first(
                                                        gradient_vector_a,
                                                        gradient_vector_b,
                                                    )
                                                    * integration_weight
                                            })
                                            .collect()
                                    })
                                    .collect(),
                                gradient_vectors
                                    .iter()
                                    .map(|gradient_vector_a| {
                                        shape_functions
                                            .iter()
                                            .map(|shape_function_b| {
                                                ((&first_piola_kirchhoff_thermal_tangent
                                                    * gradient_vector_a)
                                                    * integration_weight)
                                                    * shape_function_b
                                            })
                                            .collect()
                                    })
                                    .collect(),
                            ),
                            gradient_vectors
                                .iter()
                                .map(|gradient_vector_a| {
                                    gradient_vectors
                                        .iter()
                                        .map(|gradient_vector_b| {
                                            -(gradient_vector_a
                                                * (&heat_flux_tangent * gradient_vector_b))
                                                * integration_weight
                                        })
                                        .collect()
                                })
                                .collect(),
                        ))
                    },
                )
                .sum::<Result<_, ConstitutiveError>>()
                .map_err(|error| FiniteElementError::upstream(error, self))?;
        Ok((solid, coupling, thermal))
    }
}
