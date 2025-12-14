use crate::{
    constitutive::solid::elastic::Elastic,
    fem::{
        ElementNodalForcesSolid, ElementNodalStiffnessesSolid,
        block::element::{
            Element, ElementNodalCoordinates, FiniteElementError, solid::SolidFiniteElement,
        },
    },
    math::{ContractSecondFourthIndicesWithFirstIndicesOf, Tensor},
    mechanics::{FirstPiolaKirchhoffStresses, FirstPiolaKirchhoffTangentStiffnesses},
};
use std::fmt::Debug;

pub trait ElasticFiniteElement<C, const G: usize, const N: usize>
where
    C: Elastic,
    Self: Debug + SolidFiniteElement<G, N>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<ElementNodalForcesSolid<N>, FiniteElementError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<ElementNodalStiffnessesSolid<N>, FiniteElementError>;
}

impl<C, const G: usize, const N: usize> ElasticFiniteElement<C, G, N> for Element<G, N>
where
    C: Elastic,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<ElementNodalForcesSolid<N>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .map(|deformation_gradient| {
                constitutive_model.first_piola_kirchhoff_stress(deformation_gradient)
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
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<ElementNodalStiffnessesSolid<N>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .map(|deformation_gradient| {
                constitutive_model.first_piola_kirchhoff_tangent_stiffness(deformation_gradient)
            })
            .collect::<Result<FirstPiolaKirchhoffTangentStiffnesses<G>, _>>()
        {
            Ok(first_piola_kirchhoff_tangent_stiffnesses) => {
                Ok(first_piola_kirchhoff_tangent_stiffnesses
                    .iter()
                    .zip(
                        self.gradient_vectors()
                            .iter()
                            .zip(self.integration_weights().iter()),
                    )
                    .map(
                        |(
                            first_piola_kirchhoff_tangent_stiffness,
                            (gradient_vectors, integration_weight),
                        )| {
                            gradient_vectors
                                .iter()
                                .map(|gradient_vector_a| {
                                    gradient_vectors
                                        .iter()
                                        .map(|gradient_vector_b| {
                                            first_piola_kirchhoff_tangent_stiffness
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
