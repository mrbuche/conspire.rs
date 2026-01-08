use crate::{
    constitutive::solid::elastic_viscoplastic::ElasticViscoplastic,
    fem::block::element::{
        Element, ElementNodalCoordinates, FiniteElement, FiniteElementError,
        solid::{
            ElementNodalForcesSolid, ElementNodalStiffnessesSolid, SolidFiniteElement,
            viscoplastic::ViscoplasticStateVariables,
        },
    },
    math::{ContractSecondFourthIndicesWithFirstIndicesOf, Tensor},
    mechanics::{FirstPiolaKirchhoffStressList, FirstPiolaKirchhoffTangentStiffnessList},
};

pub trait ElasticViscoplasticFiniteElement<
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
> where
    C: ElasticViscoplastic,
    Self: SolidFiniteElement<G, M, N, P>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<ElementNodalForcesSolid<N>, FiniteElementError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<ElementNodalStiffnessesSolid<N>, FiniteElementError>;
    fn state_variables_evolution(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<ViscoplasticStateVariables<G>, FiniteElementError>;
}

impl<C, const G: usize, const N: usize, const O: usize, const P: usize>
    ElasticViscoplasticFiniteElement<C, G, 3, N, P> for Element<G, N, O>
where
    C: ElasticViscoplastic,
    Self: SolidFiniteElement<G, 3, N, P>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<ElementNodalForcesSolid<N>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(state_variables)
            .map(|(deformation_gradient, state_variable)| {
                let (deformation_gradient_p, _) = state_variable.into();
                constitutive_model
                    .first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)
            })
            .collect::<Result<FirstPiolaKirchhoffStressList<G>, _>>()
        {
            Ok(first_piola_kirchhoff_stresses) => Ok(first_piola_kirchhoff_stresses
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
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<ElementNodalStiffnessesSolid<N>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(state_variables)
            .map(|(deformation_gradient, state_variable)| {
                let (deformation_gradient_p, _) = state_variable.into();
                constitutive_model.first_piola_kirchhoff_tangent_stiffness(
                    deformation_gradient,
                    deformation_gradient_p,
                )
            })
            .collect::<Result<FirstPiolaKirchhoffTangentStiffnessList<G>, _>>()
        {
            Ok(first_piola_kirchhoff_tangent_stiffnesses) => {
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
    fn state_variables_evolution(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<ViscoplasticStateVariables<G>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(state_variables)
            .map(|(deformation_gradient, state_variable)| {
                constitutive_model.state_variables_evolution(deformation_gradient, state_variable)
            })
            .collect::<Result<ViscoplasticStateVariables<G>, _>>()
        {
            Ok(state_variables_evolution) => Ok(state_variables_evolution),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
