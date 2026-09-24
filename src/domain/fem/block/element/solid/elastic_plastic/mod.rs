use crate::{
    constitutive::{
        ConstitutiveError,
        solid::elastic_plastic::{ElasticPlastic, coupled},
    },
    domain::block::element::solid::elastic_plastic::ElasticPlasticElement,
    fem::block::element::{
        Element, ElementNodalCoordinates, FiniteElement, FiniteElementError,
        solid::{
            ElementNodalForcesSolid, ElementNodalStiffnessesSolid, SolidElement,
            SolidFiniteElement, plastic::PlasticStateVariables,
        },
    },
    math::{ContractSecondFourthWithFirst, Tensor, Vector},
    mechanics::{FirstPiolaKirchhoffStressList, FirstPiolaKirchhoffTangentStiffnessList, Scalar},
};
use std::array::from_fn;

pub trait ElasticPlasticFiniteElement<
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
> where
    C: ElasticPlastic,
    Self: SolidFiniteElement<G, M, N, P>
        + ElasticPlasticElement<
            C,
            G,
            Forces = ElementNodalForcesSolid<N>,
            Stiffnesses = ElementNodalStiffnessesSolid<N>,
            Error = FiniteElementError,
        >,
{
}

impl<T, C, const G: usize, const M: usize, const N: usize, const P: usize>
    ElasticPlasticFiniteElement<C, G, M, N, P> for T
where
    C: ElasticPlastic,
    T: SolidFiniteElement<G, M, N, P>
        + ElasticPlasticElement<
            C,
            G,
            Forces = ElementNodalForcesSolid<N>,
            Stiffnesses = ElementNodalStiffnessesSolid<N>,
            Error = FiniteElementError,
        >,
{
}

impl<C, const G: usize, const N: usize, const O: usize> ElasticPlasticElement<C, G>
    for Element<3, G, N, O>
where
    C: ElasticPlastic,
    Self: SolidFiniteElement<G, 3, N, N>,
{
    type Forces = ElementNodalForcesSolid<N>;
    type Stiffnesses = ElementNodalStiffnessesSolid<N>;
    type Error = FiniteElementError;
    fn nodal_forces_and_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &PlasticStateVariables<G>,
    ) -> Result<(ElementNodalForcesSolid<N>, ElementNodalStiffnessesSolid<N>), FiniteElementError>
    {
        let evaluations = self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(state_variables)
            .map(|(deformation_gradient, state_variable)| {
                constitutive_model
                    .condensed(deformation_gradient, state_variable)
                    .map(|(stress, tangent, _)| (stress, tangent))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| FiniteElementError::upstream(error, self))?;
        let first_piola_kirchhoff_stresses = evaluations
            .iter()
            .map(|(stress, _)| stress.clone())
            .collect::<FirstPiolaKirchhoffStressList<G>>();
        let first_piola_kirchhoff_tangent_stiffnesses = evaluations
            .into_iter()
            .map(|(_, tangent)| tangent)
            .collect::<FirstPiolaKirchhoffTangentStiffnessList<G>>();
        let forces = first_piola_kirchhoff_stresses
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
            .sum();
        let stiffnesses = first_piola_kirchhoff_tangent_stiffnesses
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
            .sum();
        Ok((forces, stiffnesses))
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
                constitutive_model
                    .condensed(deformation_gradient, state_variable)
                    .map(|(_, _, state)| state)
            })
            .collect::<Result<PlasticStateVariables<G>, _>>()
            .map_err(|error| FiniteElementError::upstream(error, self))
    }
}

/// The residuals and tangent blocks of one element of the monolithic system, in the
/// element's own numbering: `3 * N` nodal unknowns, then the local unknowns of each
/// integration point in turn. Matrices are row-major.
pub struct MonolithicElement {
    pub residual_global: Vec<Scalar>,
    pub residual_local: Vec<Scalar>,
    pub tangent_uu: Vec<Scalar>,
    pub tangent_uv: Vec<Scalar>,
    pub tangent_vu: Vec<Scalar>,
    /// One `SIZE x SIZE` block per integration point.
    pub tangent_vv: Vec<Scalar>,
}

pub trait MonolithicElasticPlasticFiniteElement<C, const G: usize, const N: usize> {
    /// Evaluates the monolithic system with the local unknowns of every integration
    /// point given as trial values, `SIZE` consecutive entries per point.
    fn monolithic(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &PlasticStateVariables<G>,
        local: &[Scalar],
    ) -> Result<MonolithicElement, FiniteElementError>;
}

impl<C, const G: usize, const N: usize, const O: usize>
    MonolithicElasticPlasticFiniteElement<C, G, N> for Element<3, G, N, O>
where
    C: ElasticPlastic,
    Self: SolidFiniteElement<G, 3, N, N>,
{
    fn monolithic(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &PlasticStateVariables<G>,
        local: &[Scalar],
    ) -> Result<MonolithicElement, FiniteElementError> {
        let (num_u, num_v) = (3 * N, coupled::SIZE * G);
        let mut element = MonolithicElement {
            residual_global: vec![0.0; num_u],
            residual_local: vec![0.0; num_v],
            tangent_uu: vec![0.0; num_u * num_u],
            tangent_uv: vec![0.0; num_u * num_v],
            tangent_vu: vec![0.0; num_v * num_u],
            tangent_vv: vec![0.0; G * coupled::SIZE * coupled::SIZE],
        };
        self.deformation_gradients(nodal_coordinates)
            .iter()
            .zip(state_variables)
            .enumerate()
            .try_for_each(|(g, (deformation_gradient, state_variable))| {
                let coupled::Monolithic {
                    stress,
                    residual_local,
                    tangent_uu: tangent,
                    tangent_vu: k_vu,
                    tangent_uv: k_uv,
                    tangent_vv: k_vv,
                } = coupled::monolithic_evaluate(
                    constitutive_model,
                    deformation_gradient,
                    state_variable,
                    &Vector::from(local[coupled::SIZE * g..coupled::SIZE * (g + 1)].to_vec()),
                )?;
                let weight = self.integration_weights()[g].value();
                let gradient: [[Scalar; 3]; N] =
                    from_fn(|a| from_fn(|j| self.gradient_vectors()[g][a][j].value()));
                (0..coupled::SIZE).for_each(|l| {
                    element.residual_local[coupled::SIZE * g + l] = residual_local[l];
                    (0..coupled::SIZE).for_each(|m| {
                        element.tangent_vv[(g * coupled::SIZE + l) * coupled::SIZE + m] = k_vv[l][m]
                    })
                });
                (0..N).for_each(|a| {
                    (0..3).for_each(|i| {
                        let row = 3 * a + i;
                        element.residual_global[row] += weight
                            * (0..3)
                                .map(|j| stress[i][j].value() * gradient[a][j])
                                .sum::<Scalar>();
                        (0..N).for_each(|b| {
                            (0..3).for_each(|k| {
                                element.tangent_uu[row * num_u + 3 * b + k] += weight
                                    * (0..3)
                                        .map(|j| {
                                            (0..3)
                                                .map(|l| {
                                                    tangent[i][j][k][l].value()
                                                        * gradient[a][j]
                                                        * gradient[b][l]
                                                })
                                                .sum::<Scalar>()
                                        })
                                        .sum::<Scalar>()
                            })
                        });
                        (0..coupled::SIZE).for_each(|l| {
                            element.tangent_uv[row * num_v + coupled::SIZE * g + l] += weight
                                * (0..3)
                                    .map(|j| k_uv[3 * i + j][l] * gradient[a][j])
                                    .sum::<Scalar>();
                            element.tangent_vu[(coupled::SIZE * g + l) * num_u + row] += (0..3)
                                .map(|j| k_vu[l][3 * i + j] * gradient[a][j])
                                .sum::<Scalar>()
                        })
                    })
                });
                Ok::<(), ConstitutiveError>(())
            })
            .map_err(|error| FiniteElementError::upstream(error, self))?;
        Ok(element)
    }
}
