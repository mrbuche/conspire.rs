use crate::{
    constitutive::solid::viscoelastic::Viscoelastic,
    fem::{
        NodalCoordinates, NodalForces, NodalStiffnesses, NodalVelocities,
        block::element::{FiniteElementError, SolidElement, SolidFiniteElement},
    },
    math::{ContractSecondFourthIndicesWithFirstIndicesOf, Tensor},
    mechanics::{FirstPiolaKirchhoffRateTangentStiffnesses, FirstPiolaKirchhoffStresses},
};

pub trait ViscoelasticFiniteElement<C, const G: usize, const N: usize>
where
    C: Viscoelastic,
    Self: SolidFiniteElement<G, N>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalForces<N>, FiniteElementError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalStiffnesses<N>, FiniteElementError>;
}

impl<C, const G: usize, const N: usize> ViscoelasticFiniteElement<C, G, N> for SolidElement<G, N>
where
    C: Viscoelastic,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalForces<N>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(
                self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                    .iter(),
            )
            .map(|(deformation_gradient, deformation_gradient_rate)| {
                constitutive_model
                    .first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_rate)
            })
            .collect::<Result<FirstPiolaKirchhoffStresses<G>, _>>()
        {
            Ok(first_piola_kirchhoff_stresses) => Ok(first_piola_kirchhoff_stresses
                .iter()
                .zip(
                    self.gradient_vectors()
                        .iter()
                        .zip(self.integration_weights().iter()),
                )
                .map(
                    |(first_piola_kirchhoff_stress, (gradient_vectors, integration_weight))| {
                        gradient_vectors
                            .iter()
                            .map(|gradient_vector| {
                                (first_piola_kirchhoff_stress * gradient_vector)
                                    * integration_weight
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
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalStiffnesses<N>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(
                self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                    .iter(),
            )
            .map(|(deformation_gradient, deformation_gradient_rate)| {
                constitutive_model.first_piola_kirchhoff_rate_tangent_stiffness(
                    deformation_gradient,
                    deformation_gradient_rate,
                )
            })
            .collect::<Result<FirstPiolaKirchhoffRateTangentStiffnesses<G>, _>>()
        {
            Ok(first_piola_kirchhoff_rate_tangent_stiffnesses) => {
                Ok(first_piola_kirchhoff_rate_tangent_stiffnesses
                    .iter()
                    .zip(
                        self.gradient_vectors().iter().zip(
                            self.gradient_vectors()
                                .iter()
                                .zip(self.integration_weights().iter()),
                        ),
                    )
                    .map(
                        |(
                            first_piola_kirchhoff_rate_tangent_stiffness,
                            (gradient_vectors_a, (gradient_vectors_b, integration_weight)),
                        )| {
                            gradient_vectors_a
                                .iter()
                                .map(|gradient_vector_a| {
                                    gradient_vectors_b
                                        .iter()
                                        .map(|gradient_vector_b| {
                                            first_piola_kirchhoff_rate_tangent_stiffness
                                        .contract_second_fourth_indices_with_first_indices_of(
                                            gradient_vector_a,
                                            gradient_vector_b,
                                        )
                                        * integration_weight
                                        })
                                        .collect()
                                })
                                .collect()
                        },
                    )
                    .sum())
            }
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
