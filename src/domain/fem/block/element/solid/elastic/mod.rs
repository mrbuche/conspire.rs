use crate::{
    constitutive::solid::elastic::Elastic,
    fem::block::element::{
        Element, ElementNodalCoordinates, FiniteElement, FiniteElementError, GradientVectors,
        solid::{ElementNodalForcesSolid, ElementNodalStiffnessesSolid, SolidFiniteElement},
        surface::{SurfaceElement, SurfaceFiniteElement},
    },
    math::{ContractSecondFourthIndicesWithFirstIndicesOf, IDENTITY, Scalar, Tensor},
    mechanics::{FirstPiolaKirchhoffStressList, FirstPiolaKirchhoffTangentStiffnessList},
};

pub trait ElasticFiniteElement<C, const G: usize, const M: usize, const N: usize, const P: usize>
where
    C: Elastic,
    Self: SolidFiniteElement<G, M, N, P>,
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

impl<C, const G: usize, const N: usize, const O: usize, const P: usize> ElasticFiniteElement<C, G, 3, N, P>
    for Element<G, N, O>
where
    C: Elastic,
    Self: SolidFiniteElement<G, 3, N, P>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<ElementNodalForcesSolid<N>, FiniteElementError> {
        nodal_forces::<_, _, _, _, _, O, _>(
            self,
            constitutive_model,
            self.gradient_vectors(),
            nodal_coordinates,
        )
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
}

impl<C, const G: usize, const N: usize, const O: usize> ElasticFiniteElement<C, G, 2, N, N>
    for SurfaceElement<G, N, O>
where
    C: Elastic,
    Self: SolidFiniteElement<G, 2, N, N>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<ElementNodalForcesSolid<N>, FiniteElementError> {
        nodal_forces::<_, _, _, _, _, O, _>(
            self,
            constitutive_model,
            self.gradient_vectors(),
            nodal_coordinates,
        )
    }
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<ElementNodalStiffnessesSolid<N>, FiniteElementError> {
        match self.deformation_gradients(nodal_coordinates).iter()
            .map(|deformation_gradient| {
                constitutive_model.first_piola_kirchhoff_tangent_stiffness(deformation_gradient)
            })
            .collect::<Result<FirstPiolaKirchhoffTangentStiffnessList<G>, _>>()
            {
            Ok(first_piola_kirchhoff_tangent_stiffnesses) => Ok(first_piola_kirchhoff_tangent_stiffnesses
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights().iter()
                    .zip(self.reference_normals().iter()
                    .zip(Self::normal_gradients(nodal_coordinates))
                )
                ),
            )
            .map(
                |(
                    first_piola_kirchhoff_tangent_stiffness,
                    (gradient_vectors, (integration_weight, (reference_normal, normal_gradients))),
                )| {
                    gradient_vectors.iter()
                    .map(|gradient_vector_a|
                        gradient_vectors.iter()
                        .zip(normal_gradients.iter())
                        .map(|(gradient_vector_b, normal_gradient_b)|
                            first_piola_kirchhoff_tangent_stiffness.iter()
                            .map(|first_piola_kirchhoff_tangent_stiffness_m|
                                IDENTITY.iter()
                                .zip(normal_gradient_b.iter())
                                .map(|(identity_n, normal_gradient_b_n)|
                                    first_piola_kirchhoff_tangent_stiffness_m.iter()
                                    .zip(gradient_vector_a.iter())
                                    .map(|(first_piola_kirchhoff_tangent_stiffness_mj, gradient_vector_a_j)|
                                        first_piola_kirchhoff_tangent_stiffness_mj.iter()
                                        .zip(identity_n.iter()
                                        .zip(normal_gradient_b_n.iter()))
                                        .map(|(first_piola_kirchhoff_tangent_stiffness_mjk, (identity_nk, normal_gradient_b_n_k))|
                                            first_piola_kirchhoff_tangent_stiffness_mjk.iter()
                                            .zip(gradient_vector_b.iter()
                                            .zip(reference_normal.iter()))
                                            .map(|(first_piola_kirchhoff_tangent_stiffness_mjkl, (gradient_vector_b_l, reference_normal_l))|
                                                first_piola_kirchhoff_tangent_stiffness_mjkl * gradient_vector_a_j * (
                                                    identity_nk * gradient_vector_b_l + normal_gradient_b_n_k * reference_normal_l
                                                ) * integration_weight
                                            ).sum::<Scalar>()
                                        ).sum::<Scalar>()
                                    ).sum::<Scalar>()
                                ).collect()
                            ).collect()
                        ).collect()
                    ).collect()
                }
            )
            .sum()),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

fn nodal_forces<C, F, const G: usize, const M: usize, const N: usize, const O: usize, const P: usize>(
    element: &F,
    constitutive_model: &C,
    gradient_vectors: &GradientVectors<G, N>,
    nodal_coordinates: &ElementNodalCoordinates<N>,
) -> Result<ElementNodalForcesSolid<N>, FiniteElementError>
where
    C: Elastic,
    F: SolidFiniteElement<G, M, N, P>,
{
    match element
        .deformation_gradients(nodal_coordinates)
        .iter()
        .map(|deformation_gradient| {
            constitutive_model.first_piola_kirchhoff_stress(deformation_gradient)
        })
        .collect::<Result<FirstPiolaKirchhoffStressList<G>, _>>()
    {
        Ok(first_piola_kirchhoff_stresses) => Ok(first_piola_kirchhoff_stresses
            .iter()
            .zip(gradient_vectors.iter().zip(element.integration_weights()))
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
            .sum()),
        Err(error) => Err(FiniteElementError::Upstream(
            format!("{error}"),
            format!("{element:?}"),
        )),
    }
}
