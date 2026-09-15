use crate::{
    constitutive::{
        ConstitutiveError,
        solid::elastic_plastic::{ElasticPlastic, fischer_burmeister},
    },
    fem::block::element::{
        Element, ElementNodalCoordinates, FiniteElement, FiniteElementError,
        solid::{
            ElementNodalForcesSolid, ElementNodalStiffnessesSolid, SolidFiniteElement,
            plastic::PlasticStateVariables,
        },
    },
    math::{ContractSecondFourthWithFirst, Quantity, Rank2, Scalar, Tensor},
    mechanics::{
        DeformationGradientPlastic, FirstPiolaKirchhoffStressList,
        FirstPiolaKirchhoffTangentStiffnessList,
    },
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
    /// Residual and tangent contributions for the monolithic (block) solve, with the
    /// plastic multiplier at every integration point a free unknown of the outer solver
    /// rather than condensed out by [`Self::nodal_forces`]/[`Self::nodal_stiffnesses`].
    ///
    /// `state_variables` is the previously converged plastic state (fixes the flow
    /// direction reference and the equivalent plastic strain to step from);
    /// `plastic_multipliers` is the current trial `\Delta\gamma` at each point. Returns
    /// `(residual_global, residual_local, K_uu, K_uv, K_vu, K_vv)`, with `K_uv`/`K_vu`
    /// indexed `[point][node][component]` since neither has an existing typed shape.
    #[expect(clippy::type_complexity)]
    fn monolithic_contributions(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &PlasticStateVariables<G>,
        plastic_multipliers: &[Scalar; G],
    ) -> Result<
        (
            ElementNodalForcesSolid<N>,
            [Scalar; G],
            ElementNodalStiffnessesSolid<N>,
            [[[Scalar; 3]; N]; G],
            [[[Scalar; 3]; N]; G],
            [Scalar; G],
        ),
        FiniteElementError,
    >;
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
    fn monolithic_contributions(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &PlasticStateVariables<G>,
        plastic_multipliers: &[Scalar; G],
    ) -> Result<
        (
            ElementNodalForcesSolid<N>,
            [Scalar; G],
            ElementNodalStiffnessesSolid<N>,
            [[[Scalar; 3]; N]; G],
            [[[Scalar; 3]; N]; G],
            [Scalar; G],
        ),
        FiniteElementError,
    > {
        let deformation_gradients = self.deformation_gradients(nodal_coordinates);
        let per_point = deformation_gradients
            .iter()
            .zip(state_variables)
            .zip(plastic_multipliers)
            .map(
                |((deformation_gradient, state_variable), &plastic_multiplier)| {
                    let (
                        plastic_deformation_gradient_previous,
                        &equivalent_plastic_strain_previous,
                    ): (&DeformationGradientPlastic, &Quantity) = state_variable.into();
                    let flow_direction = {
                        let deviatoric = constitutive_model
                            .mandel_stress(
                                deformation_gradient,
                                plastic_deformation_gradient_previous,
                            )?
                            .deviatoric();
                        let direction = constitutive_model.flow_direction(&deviatoric)?;
                        (&direction + direction.transpose()) * 0.5
                    };
                    let plastic =
                        (&flow_direction * plastic_multiplier)
                            .expm()
                            .map_err(|error| {
                                ConstitutiveError::custom(format!("{error:?}"), constitutive_model)
                            })?
                            * plastic_deformation_gradient_previous;
                    let stress = constitutive_model
                        .first_piola_kirchhoff_stress(deformation_gradient, &plastic)?;
                    let deviatoric = constitutive_model
                        .mandel_stress(deformation_gradient, &plastic)?
                        .deviatoric();
                    let scaled = constitutive_model
                        .yield_function(
                            &deviatoric,
                            equivalent_plastic_strain_previous + Quantity::new(plastic_multiplier),
                        )?
                        .value()
                        / constitutive_model.initial_yield_stress().value();
                    let residual_local = fischer_burmeister(plastic_multiplier, -scaled);
                    let (tangent, k_vu, k_uv, k_vv) = constitutive_model.monolithic_tangents(
                        deformation_gradient,
                        plastic_deformation_gradient_previous,
                        &flow_direction,
                        equivalent_plastic_strain_previous,
                        plastic_multiplier,
                    )?;
                    Ok::<_, crate::constitutive::ConstitutiveError>((
                        stress,
                        residual_local,
                        tangent,
                        k_uv,
                        k_vu.0,
                        k_vv.value(),
                    ))
                },
            )
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| FiniteElementError::upstream(error, self))?;
        let residual_local = std::array::from_fn(|g| per_point[g].1);
        let k_vv = std::array::from_fn(|g| per_point[g].5);
        let residual_global = per_point
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(|((stress, ..), (gradient_vectors, integration_weight))| {
                gradient_vectors
                    .iter()
                    .map(|gradient_vector| (stress * gradient_vector) * integration_weight)
                    .collect()
            })
            .sum();
        let k_uu = per_point
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(
                |((_, _, tangent, ..), (gradient_vectors, integration_weight))| {
                    gradient_vectors
                        .iter()
                        .map(|gradient_vector_a| {
                            gradient_vectors
                                .iter()
                                .map(|gradient_vector_b| {
                                    tangent.contract_second_fourth_with_first(
                                        gradient_vector_a,
                                        gradient_vector_b,
                                    ) * integration_weight
                                })
                                .collect()
                        })
                        .collect()
                },
            )
            .sum();
        let k_uv = std::array::from_fn(|g| {
            let (_, _, _, k_uv_g, _, _) = &per_point[g];
            let (gradient_vectors, integration_weight) =
                (&self.gradient_vectors()[g], &self.integration_weights()[g]);
            std::array::from_fn(|a| {
                let contribution = (k_uv_g * &gradient_vectors[a]) * integration_weight;
                [
                    contribution[0].value(),
                    contribution[1].value(),
                    contribution[2].value(),
                ]
            })
        });
        let k_vu = std::array::from_fn(|g| {
            let (_, _, _, _, k_vu_g, _) = &per_point[g];
            let gradient_vectors = &self.gradient_vectors()[g];
            std::array::from_fn(|a| {
                let contribution = k_vu_g * &gradient_vectors[a];
                [
                    contribution[0].value(),
                    contribution[1].value(),
                    contribution[2].value(),
                ]
            })
        });
        Ok((residual_global, residual_local, k_uu, k_uv, k_vu, k_vv))
    }
}
