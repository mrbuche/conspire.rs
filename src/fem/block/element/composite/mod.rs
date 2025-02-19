#[cfg(test)]
mod test;

pub mod tetrahedron;

use super::*;

pub trait CompositeElement<
    'a,
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    const Q: usize,
> where
    C: Constitutive<'a>,
{
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> DeformationGradients<G> {
        self.get_projected_gradient_vectors()
            .iter()
            .map(|projected_gradient_vectors| {
                nodal_coordinates
                    .iter()
                    .zip(projected_gradient_vectors.iter())
                    .map(|(nodal_coordinate, projected_gradient_vector)| {
                        DeformationGradient::dyad(nodal_coordinate, projected_gradient_vector)
                    })
                    .sum()
            })
            .collect()
    }
    fn deformation_gradient_rates(
        &self,
        _: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> DeformationGradientRates<G> {
        self.get_projected_gradient_vectors()
            .iter()
            .map(|projected_gradient_vectors| {
                nodal_velocities
                    .iter()
                    .zip(projected_gradient_vectors.iter())
                    .map(|(nodal_velocity, projected_gradient_vector)| {
                        DeformationGradientRate::dyad(nodal_velocity, projected_gradient_vector)
                    })
                    .sum()
            })
            .collect()
    }
    fn inverse_normalized_projection_matrix() -> NormalizedProjectionMatrix<Q>;
    fn inverse_projection_matrix(
        reference_jacobians_subelements: &Scalars<P>,
    ) -> NormalizedProjectionMatrix<Q> {
        Self::shape_function_integrals_products()
            .iter()
            .zip(reference_jacobians_subelements.iter())
            .map(
                |(shape_function_integrals_products, reference_jacobian_subelement)| {
                    shape_function_integrals_products * reference_jacobian_subelement
                },
            )
            .sum::<ProjectionMatrix<Q>>()
            .inverse()
    }
    fn projected_gradient_vectors(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
    ) -> ProjectedGradientVectors<G, N>;
    fn reference_jacobians(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
    ) -> Scalars<G> {
        let vector = Self::inverse_normalized_projection_matrix()
            * Self::shape_function_integrals()
                .iter()
                .zip(Self::reference_jacobians_subelements(reference_nodal_coordinates).iter())
                .map(|(shape_function_integral, reference_jacobian_subelement)| {
                    shape_function_integral * reference_jacobian_subelement
                })
                .sum::<TensorRank1<Q, 9>>();
        Self::shape_functions_at_integration_points()
            .iter()
            .map(|shape_functions_at_integration_point| {
                shape_functions_at_integration_point * &vector
            })
            .collect()
    }
    fn reference_jacobians_subelements(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
    ) -> Scalars<P>;
    fn shape_function_integrals() -> ShapeFunctionIntegrals<P, Q>;
    fn shape_function_integrals_products() -> ShapeFunctionIntegralsProducts<P, Q>;
    fn shape_functions_at_integration_points() -> ShapeFunctionsAtIntegrationPoints<G, Q>;
    fn standard_gradient_operators() -> StandardGradientOperators<M, O, P>;
    fn standard_gradient_operators_transposed() -> StandardGradientOperatorsTransposed<M, O, P>;
    fn get_constitutive_models(&self) -> &[C; G];
    fn get_integration_weights(&self) -> &Scalars<G>;
    fn get_projected_gradient_vectors(&self) -> &ProjectedGradientVectors<G, N>;
}

pub trait ElasticCompositeElement<
    'a,
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    const Q: usize,
> where
    C: Elastic<'a>,
    Self: CompositeElement<'a, C, G, M, N, O, P, Q>,
{
    fn nodal_forces_composite_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalForces<N>, ConstitutiveError> {
        Ok(self
            .get_constitutive_models()
            .iter()
            .zip(self.deformation_gradients(nodal_coordinates).iter())
            .map(|(constitutive_model, deformation_gradient)| {
                constitutive_model.first_piola_kirchhoff_stress(deformation_gradient)
            })
            .collect::<Result<FirstPiolaKirchhoffStresses<G>, _>>()?
            .iter()
            .zip(
                self.get_projected_gradient_vectors()
                    .iter()
                    .zip(self.get_integration_weights().iter()),
            )
            .map(
                |(
                    first_piola_kirchhoff_stress,
                    (projected_gradient_vectors, scaled_composite_jacobian),
                )| {
                    projected_gradient_vectors
                        .iter()
                        .map(|projected_gradient_vector| {
                            (first_piola_kirchhoff_stress * projected_gradient_vector)
                                * scaled_composite_jacobian
                        })
                        .collect()
                },
            )
            .sum())
    }
    fn nodal_stiffnesses_composite_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalStiffnesses<N>, ConstitutiveError> {
        Ok(self
            .get_constitutive_models()
            .iter()
            .zip(self.deformation_gradients(nodal_coordinates).iter())
            .map(|(constitutive_model, deformation_gradient)| {
                constitutive_model.first_piola_kirchhoff_tangent_stiffness(deformation_gradient)
            })
            .collect::<Result<FirstPiolaKirchhoffTangentStiffnesses<G>, _>>()?
            .iter()
            .zip(
                self.get_projected_gradient_vectors()
                    .iter()
                    .zip(self.get_integration_weights().iter()),
            )
            .map(
                |(
                    first_piola_kirchhoff_tangent_stiffness,
                    (projected_gradient_vectors, scaled_composite_jacobian),
                )| {
                    projected_gradient_vectors
                        .iter()
                        .map(|projected_gradient_vector_a| {
                            projected_gradient_vectors
                                .iter()
                                .map(|projected_gradient_vector_b| {
                                    first_piola_kirchhoff_tangent_stiffness
                                        .contract_second_fourth_indices_with_first_indices_of(
                                            projected_gradient_vector_a,
                                            projected_gradient_vector_b,
                                        )
                                        * scaled_composite_jacobian
                                })
                                .collect()
                        })
                        .collect()
                },
            )
            .sum())
    }
}

pub trait HyperelasticCompositeElement<
    'a,
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    const Q: usize,
> where
    C: Hyperelastic<'a>,
    Self: ElasticCompositeElement<'a, C, G, M, N, O, P, Q>,
{
    fn helmholtz_free_energy_composite_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.get_constitutive_models()
            .iter()
            .zip(
                self.deformation_gradients(nodal_coordinates)
                    .iter()
                    .zip(self.get_integration_weights().iter()),
            )
            .map(
                |(constitutive_model, (deformation_gradient, scaled_composite_jacobian))| {
                    Ok(
                        constitutive_model.helmholtz_free_energy_density(deformation_gradient)?
                            * scaled_composite_jacobian,
                    )
                },
            )
            .sum()
    }
}

pub trait ViscoelasticCompositeElement<
    'a,
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    const Q: usize,
> where
    C: Viscoelastic<'a>,
    Self: CompositeElement<'a, C, G, M, N, O, P, Q>,
{
    fn nodal_forces_composite_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalForces<N>, ConstitutiveError> {
        Ok(self
            .get_constitutive_models()
            .iter()
            .zip(
                self.deformation_gradients(nodal_coordinates).iter().zip(
                    self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                        .iter(),
                ),
            )
            .map(
                |(constitutive_model, (deformation_gradient, deformation_gradient_rate))| {
                    constitutive_model.first_piola_kirchhoff_stress(
                        deformation_gradient,
                        deformation_gradient_rate,
                    )
                },
            )
            .collect::<Result<FirstPiolaKirchhoffStresses<G>, _>>()?
            .iter()
            .zip(
                self.get_projected_gradient_vectors()
                    .iter()
                    .zip(self.get_integration_weights().iter()),
            )
            .map(
                |(
                    first_piola_kirchhoff_stress,
                    (projected_gradient_vectors, scaled_composite_jacobian),
                )| {
                    projected_gradient_vectors
                        .iter()
                        .map(|projected_gradient_vector| {
                            (first_piola_kirchhoff_stress * projected_gradient_vector)
                                * scaled_composite_jacobian
                        })
                        .collect()
                },
            )
            .sum())
    }
    fn nodal_stiffnesses_composite_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalStiffnesses<N>, ConstitutiveError> {
        Ok(self
            .get_constitutive_models()
            .iter()
            .zip(
                self.deformation_gradients(nodal_coordinates).iter().zip(
                    self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                        .iter(),
                ),
            )
            .map(
                |(constitutive_model, (deformation_gradient, deformation_gradient_rate))| {
                    constitutive_model.first_piola_kirchhoff_rate_tangent_stiffness(
                        deformation_gradient,
                        deformation_gradient_rate,
                    )
                },
            )
            .collect::<Result<FirstPiolaKirchhoffRateTangentStiffnesses<G>, _>>()?
            .iter()
            .zip(
                self.get_projected_gradient_vectors().iter().zip(
                    self.get_projected_gradient_vectors()
                        .iter()
                        .zip(self.get_integration_weights().iter()),
                ),
            )
            .map(
                |(
                    first_piola_kirchhoff_rate_tangent_stiffness,
                    (
                        projected_gradient_vectors_a,
                        (projected_gradient_vectors_b, scaled_composite_jacobian),
                    ),
                )| {
                    projected_gradient_vectors_a
                        .iter()
                        .map(|projected_gradient_vector_a| {
                            projected_gradient_vectors_b
                                .iter()
                                .map(|projected_gradient_vector_b| {
                                    first_piola_kirchhoff_rate_tangent_stiffness
                                        .contract_second_fourth_indices_with_first_indices_of(
                                            projected_gradient_vector_a,
                                            projected_gradient_vector_b,
                                        )
                                        * scaled_composite_jacobian
                                })
                                .collect()
                        })
                        .collect()
                },
            )
            .sum())
    }
}

pub trait ElasticHyperviscousCompositeElement<
    'a,
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    const Q: usize,
> where
    C: ElasticHyperviscous<'a>,
    Self: ViscoelasticCompositeElement<'a, C, G, M, N, O, P, Q>,
{
    fn viscous_dissipation_composite_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.get_constitutive_models()
            .iter()
            .zip(
                self.deformation_gradients(nodal_coordinates).iter().zip(
                    self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                        .iter()
                        .zip(self.get_integration_weights().iter()),
                ),
            )
            .map(
                |(
                    constitutive_model,
                    (deformation_gradient, (deformation_gradient_rate, scaled_composite_jacobian)),
                )| {
                    Ok(constitutive_model
                        .viscous_dissipation(deformation_gradient, deformation_gradient_rate)?
                        * scaled_composite_jacobian)
                },
            )
            .sum()
    }
    fn dissipation_potential_composite_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.get_constitutive_models()
            .iter()
            .zip(
                self.deformation_gradients(nodal_coordinates).iter().zip(
                    self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                        .iter()
                        .zip(self.get_integration_weights().iter()),
                ),
            )
            .map(
                |(
                    constitutive_model,
                    (deformation_gradient, (deformation_gradient_rate, scaled_composite_jacobian)),
                )| {
                    Ok(constitutive_model
                        .dissipation_potential(deformation_gradient, deformation_gradient_rate)?
                        * scaled_composite_jacobian)
                },
            )
            .sum()
    }
}

pub trait HyperviscoelasticCompositeElement<
    'a,
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    const Q: usize,
> where
    C: Hyperviscoelastic<'a>,
    Self: ElasticHyperviscousCompositeElement<'a, C, G, M, N, O, P, Q>,
{
    fn helmholtz_free_energy_composite_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.get_constitutive_models()
            .iter()
            .zip(
                self.deformation_gradients(nodal_coordinates)
                    .iter()
                    .zip(self.get_integration_weights().iter()),
            )
            .map(
                |(constitutive_model, (deformation_gradient, scaled_composite_jacobian))| {
                    Ok(
                        constitutive_model.helmholtz_free_energy_density(deformation_gradient)?
                            * scaled_composite_jacobian,
                    )
                },
            )
            .sum()
    }
}
