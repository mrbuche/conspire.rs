use crate::{
    constitutive::solid::elastic_plastic::ElasticPlastic,
    fem::block::element::{
        Element, ElementNodalCoordinates, FiniteElement, FiniteElementError,
        solid::{
            ElementNodalForcesSolid, ElementNodalStiffnessesSolid, SolidFiniteElement,
            plastic::PlasticStateVariables,
        },
    },
    math::{ContractSecondFourthWithFirst, Tensor},
    mechanics::{FirstPiolaKirchhoffStressList, FirstPiolaKirchhoffTangentStiffnessList},
};

pub trait ElasticPlasticFiniteElement<
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
> where
    C: ElasticPlastic,
    Self: SolidFiniteElement<G, M, N, P>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &PlasticStateVariables<G>,
    ) -> Result<ElementNodalForcesSolid<N>, FiniteElementError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &PlasticStateVariables<G>,
    ) -> Result<ElementNodalStiffnessesSolid<N>, FiniteElementError>;
    fn updated_state(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &PlasticStateVariables<G>,
    ) -> Result<PlasticStateVariables<G>, FiniteElementError>;
}

impl<C, const G: usize, const N: usize, const O: usize, const P: usize>
    ElasticPlasticFiniteElement<C, G, 3, N, P> for Element<3, G, N, O>
where
    C: ElasticPlastic,
    Self: SolidFiniteElement<G, 3, N, P>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &PlasticStateVariables<G>,
    ) -> Result<ElementNodalForcesSolid<N>, FiniteElementError> {
        let first_piola_kirchhoff_stresses = self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(state_variables)
            .map(|(deformation_gradient, state_variable)| {
                let updated =
                    constitutive_model.return_map(deformation_gradient, state_variable)?;
                constitutive_model.first_piola_kirchhoff_stress(deformation_gradient, &updated.0)
            })
            .collect::<Result<FirstPiolaKirchhoffStressList<G>, _>>()
            .map_err(|error| FiniteElementError::upstream(error, self))?;
        Ok(first_piola_kirchhoff_stresses
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(
                |(first_piola_kirchhoff_stress, (gradient_vectors, integration_weight))| {
                    gradient_vectors
                        .iter()
                        .map(|gradient_vector| {
                            (first_piola_kirchhoff_stress * gradient_vector) * integration_weight
                        })
                        .collect()
                },
            )
            .sum())
    }
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &PlasticStateVariables<G>,
    ) -> Result<ElementNodalStiffnessesSolid<N>, FiniteElementError> {
        let first_piola_kirchhoff_tangent_stiffnesses = self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(state_variables)
            .map(|(deformation_gradient, state_variable)| {
                constitutive_model
                    .consistent_tangent_stiffness(deformation_gradient, state_variable)
                    .map(|(tangent, _)| tangent)
            })
            .collect::<Result<FirstPiolaKirchhoffTangentStiffnessList<G>, _>>()
            .map_err(|error| FiniteElementError::upstream(error, self))?;
        Ok(first_piola_kirchhoff_tangent_stiffnesses
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights()),
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
                                        .contract_second_fourth_with_first(
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
    fn updated_state(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &PlasticStateVariables<G>,
    ) -> Result<PlasticStateVariables<G>, FiniteElementError> {
        self.deformation_gradients(nodal_coordinates)
            .iter()
            .zip(state_variables)
            .map(|(deformation_gradient, state_variable)| {
                constitutive_model.return_map(deformation_gradient, state_variable)
            })
            .collect::<Result<PlasticStateVariables<G>, _>>()
            .map_err(|error| FiniteElementError::upstream(error, self))
    }
}
