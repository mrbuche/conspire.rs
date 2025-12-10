use crate::{
    constitutive::thermal::conduction::ThermalConduction,
    fem::{
        NodalForcesThermal, NodalStiffnessesThermal, NodalTemperatures,
        block::element::{Element, FiniteElementError, thermal::ThermalFiniteElement},
    },
    math::Tensor,
    mechanics::HeatFluxes,
};

pub trait ThermalConductionFiniteElement<C, const G: usize, const N: usize>
where
    C: ThermalConduction,
    Self: ThermalFiniteElement<G, N>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &NodalTemperatures<N>,
    ) -> Result<NodalForcesThermal<N>, FiniteElementError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &NodalTemperatures<N>,
    ) -> Result<NodalStiffnessesThermal<N>, FiniteElementError>;
}

impl<C, const G: usize, const N: usize> ThermalConductionFiniteElement<C, G, N> for Element<G, N>
where
    C: ThermalConduction,
    Self: ThermalFiniteElement<G, N>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &NodalTemperatures<N>,
    ) -> Result<NodalForcesThermal<N>, FiniteElementError> {
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
                        .zip(self.integration_weights().iter()),
                )
                .map(|(heat_flux, (gradient_vectors, integration_weight))| {
                    gradient_vectors
                        .iter()
                        .map(|gradient_vector| (heat_flux * gradient_vector) * integration_weight)
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
        nodal_temperatures: &NodalTemperatures<N>,
    ) -> Result<NodalStiffnessesThermal<N>, FiniteElementError> {
        todo!()
        // match self
        //     .deformation_gradients(nodal_coordinates)
        //     .iter()
        //     .map(|deformation_gradient| {
        //         constitutive_model.first_piola_kirchhoff_tangent_stiffness(deformation_gradient)
        //     })
        //     .collect::<Result<FirstPiolaKirchhoffTangentStiffnesses<G>, _>>()
        // {
        //     Ok(first_piola_kirchhoff_tangent_stiffnesses) => {
        //         Ok(first_piola_kirchhoff_tangent_stiffnesses
        //             .iter()
        //             .zip(
        //                 self.gradient_vectors()
        //                     .iter()
        //                     .zip(self.integration_weights().iter()),
        //             )
        //             .map(
        //                 |(
        //                     first_piola_kirchhoff_tangent_stiffness,
        //                     (gradient_vectors, integration_weight),
        //                 )| {
        //                     gradient_vectors
        //                         .iter()
        //                         .map(|gradient_vector_a| {
        //                             gradient_vectors
        //                                 .iter()
        //                                 .map(|gradient_vector_b| {
        //                                     first_piola_kirchhoff_tangent_stiffness
        //                                     .contract_second_fourth_indices_with_first_indices_of(
        //                                         gradient_vector_a,
        //                                         gradient_vector_b,
        //                                     )
        //                                     * integration_weight
        //                                 })
        //                                 .collect()
        //                         })
        //                         .collect()
        //                 },
        //             )
        //             .sum())
        //     }
        //     Err(error) => Err(FiniteElementError::Upstream(
        //         format!("{error}"),
        //         format!("{self:?}"),
        //     )),
        // }
    }
}
