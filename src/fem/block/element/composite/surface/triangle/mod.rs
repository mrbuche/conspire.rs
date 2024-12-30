#[cfg(test)]
mod test;

use super::*;

const G: usize = 3;
const M: usize = 2;
const N: usize = 6;
const O: usize = 6;
const P: usize = 4;
const Q: usize = 3;

const INTEGRATION_WEIGHT: Scalar = ONE_SIXTH;

pub const INVERSE_NORMALIED_PROJECTION_MATRIX: NormalizedProjectionMatrix<Q> =
    TensorRank2::<Q, 9, 9>([
        TensorRank1([3.0 / 64.0, -1.0 / 64.0, -1.0 / 64.0]),
        TensorRank1([-1.0 / 64.0, 3.0 / 64.0, -1.0 / 64.0]),
        TensorRank1([-1.0 / 64.0, -1.0 / 64.0, 3.0 / 64.0]),
    ]);
pub const SHAPE_FUNCTION_INTEGRALS: ShapeFunctionIntegrals<P, Q> = TensorRank1List::<Q, 9, P>([
    TensorRank1([32.0, 8.0, 8.0]),
    TensorRank1([8.0, 32.0, 8.0]),
    TensorRank1([8.0, 8.0, 32.0]),
    TensorRank1([16.0, 16.0, 16.0]),
]);
pub const SHAPE_FUNCTION_INTEGRALS_PRODUCTS: ShapeFunctionIntegralsProducts<P, Q> =
    TensorRank2List::<Q, 9, 9, P>([
        TensorRank2([
            TensorRank1([22.0, 5.0, 5.0]),
            TensorRank1([5.0, 2.0, 1.0]),
            TensorRank1([5.0, 1.0, 2.0]),
        ]),
        TensorRank2([
            TensorRank1([2.0, 5.0, 1.0]),
            TensorRank1([5.0, 22.0, 5.0]),
            TensorRank1([1.0, 5.0, 2.0]),
        ]),
        TensorRank2([
            TensorRank1([2.0, 1.0, 5.0]),
            TensorRank1([1.0, 2.0, 5.0]),
            TensorRank1([5.0, 5.0, 22.0]),
        ]),
        TensorRank2([
            TensorRank1([6.0, 5.0, 5.0]),
            TensorRank1([5.0, 6.0, 5.0]),
            TensorRank1([5.0, 5.0, 6.0]),
        ]),
    ]);
const DIAG: Scalar = 0.666_666_666_666_666_6;
const OFF: Scalar = 0.166_666_666_666_666_7;
pub const SHAPE_FUNCTIONS_AT_INTEGRATION_POINTS: ShapeFunctionsAtIntegrationPoints<G, Q> =
    TensorRank1List::<Q, 9, G>([
        TensorRank1([DIAG, OFF, OFF]),
        TensorRank1([OFF, DIAG, OFF]),
        TensorRank1([OFF, OFF, DIAG]),
    ]);
pub const STANDARD_GRADIENT_OPERATORS: StandardGradientOperators<M, O, P> =
    TensorRank1List2D::<M, 9, O, P>([
        TensorRank1List([
            TensorRank1([2.0, 0.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([0.0, 2.0]),
            tensor_rank_1_zero(),
            TensorRank1([-2.0, -2.0]),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            TensorRank1([0.0, 2.0]),
            tensor_rank_1_zero(),
            TensorRank1([2.0, 0.0]),
            TensorRank1([-2.0, -2.0]),
            tensor_rank_1_zero(),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([-2.0, -2.0]),
            tensor_rank_1_zero(),
            TensorRank1([0.0, 2.0]),
            TensorRank1([2.0, 0.0]),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([2.0, 2.0]),
            TensorRank1([-2.0, 0.0]),
            TensorRank1([0.0, -2.0]),
        ]),
    ]);

pub const STANDARD_GRADIENT_OPERATORS_TRANSPOSED: StandardGradientOperatorsTransposed<M, O, P> =
    TensorRank1List2D::<M, 9, P, O>([
        TensorRank1List([
            TensorRank1([2.0, 0.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            TensorRank1([0.0, 2.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([-2.0, -2.0]),
            tensor_rank_1_zero(),
        ]),
        TensorRank1List([
            TensorRank1([0.0, 2.0]),
            TensorRank1([2.0, 0.0]),
            tensor_rank_1_zero(),
            TensorRank1([2.0, 2.0]),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            TensorRank1([-2.0, -2.0]),
            TensorRank1([0.0, 2.0]),
            TensorRank1([-2.0, 0.0]),
        ]),
        TensorRank1List([
            TensorRank1([-2.0, -2.0]),
            tensor_rank_1_zero(),
            TensorRank1([2.0, 0.0]),
            TensorRank1([0.0, -2.0]),
        ]),
    ]);

pub struct Triangle<C> {
    constitutive_models: [C; G],
    integration_weights: Scalars<G>,
    projected_gradient_vectors: ProjectedGradientVectors<G, N>,
    scaled_reference_normals: ScaledReferenceNormals<G, P>,
}

impl<'a, C> SurfaceElement<'a, C, G, N> for Triangle<C>
where
    C: Constitutive<'a>,
{
    fn new(
        constitutive_model_parameters: Parameters<'a>,
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
        thickness: &Scalar,
    ) -> Self {
        Self {
            constitutive_models: std::array::from_fn(|_| <C>::new(constitutive_model_parameters)),
            integration_weights: Self::calculate_reference_jacobians(&reference_nodal_coordinates)
                * (INTEGRATION_WEIGHT * thickness),
            projected_gradient_vectors: Self::calculate_projected_gradient_vectors(
                &reference_nodal_coordinates,
            ),
            scaled_reference_normals: Self::calculate_scaled_reference_normals(
                &reference_nodal_coordinates,
            ),
        }
    }
}

impl<'a, C> CompositeElement<'a, C, G, M, N, O, P, Q> for Triangle<C>
where
    C: Constitutive<'a>,
{
    fn calculate_deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> DeformationGradients<G> {
        self.calculate_deformation_gradients_composite_surface_element(nodal_coordinates)
    }
    fn calculate_deformation_gradient_rates(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> DeformationGradientRates<G> {
        self.calculate_deformation_gradient_rates_composite_surface_element(
            nodal_coordinates,
            nodal_velocities,
        )
    }
    fn calculate_inverse_normalized_projection_matrix() -> NormalizedProjectionMatrix<Q> {
        INVERSE_NORMALIED_PROJECTION_MATRIX
    }
    fn calculate_projected_gradient_vectors(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
    ) -> ProjectedGradientVectors<G, N> {
        Self::calculate_projected_gradient_vectors_composite_surface_element(
            reference_nodal_coordinates,
        )
    }
    fn calculate_reference_jacobians_subelements(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
    ) -> Scalars<P> {
        Self::calculate_bases(reference_nodal_coordinates)
            .iter()
            .map(|reference_basis_vectors| {
                reference_basis_vectors[0]
                    .cross(&reference_basis_vectors[1])
                    .norm()
            })
            .collect()
    }
    fn calculate_shape_function_integrals() -> ShapeFunctionIntegrals<P, Q> {
        SHAPE_FUNCTION_INTEGRALS
    }
    fn calculate_shape_function_integrals_products() -> ShapeFunctionIntegralsProducts<P, Q> {
        SHAPE_FUNCTION_INTEGRALS_PRODUCTS
    }
    fn calculate_shape_functions_at_integration_points() -> ShapeFunctionsAtIntegrationPoints<G, Q>
    {
        SHAPE_FUNCTIONS_AT_INTEGRATION_POINTS
    }
    fn calculate_standard_gradient_operators() -> StandardGradientOperators<M, O, P> {
        STANDARD_GRADIENT_OPERATORS
    }
    fn calculate_standard_gradient_operators_transposed(
    ) -> StandardGradientOperatorsTransposed<M, O, P> {
        STANDARD_GRADIENT_OPERATORS_TRANSPOSED
    }
    fn get_constitutive_models(&self) -> &[C; G] {
        &self.constitutive_models
    }
    fn get_integration_weights(&self) -> &Scalars<G> {
        &self.integration_weights
    }
    fn get_projected_gradient_vectors(&self) -> &ProjectedGradientVectors<G, N> {
        &self.projected_gradient_vectors
    }
}

impl<'a, C> ElasticFiniteElement<'a, C, G, N> for Triangle<C>
where
    C: Elastic<'a>,
{
    fn calculate_nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalForces<N>, ConstitutiveError> {
        Ok(self.get_constitutive_models().iter()
        .zip(self.calculate_deformation_gradients(nodal_coordinates).iter())
        .map(|(constitutive_model, deformation_gradient)|
            constitutive_model.calculate_first_piola_kirchoff_stress(deformation_gradient)
        ).collect::<Result<FirstPiolaKirchoffStresses<G>, _>>()?.iter()
        .zip(self.get_projected_gradient_vectors().iter()
        .zip(self.get_integration_weights().iter()
        .zip(self.calculate_objects(&Self::calculate_normal_gradients(nodal_coordinates)).iter())))
        .map(|(first_piola_kirchoff_stress, (projected_gradient_vectors, (scaled_composite_jacobian, objects)))|
            projected_gradient_vectors.iter()
            .zip(objects.iter())
            .map(|(projected_gradient_vector, object)|
                IDENTITY.iter()
                .zip(object.iter())
                .map(|(identity_m, object_m)|
                    first_piola_kirchoff_stress.iter()
                    .zip(identity_m.iter()
                    .zip(object_m.iter()))
                    .map(|(first_piola_kirchoff_stress_i, (identity_mi, object_mi))|
                        first_piola_kirchoff_stress_i.iter()
                        .zip(projected_gradient_vector.iter()
                        .zip(object_mi.iter()))
                        .map(|(first_piola_kirchoff_stress_ij, (projected_gradient_vector_j, object_mij))|
                            first_piola_kirchoff_stress_ij * (
                                identity_mi * projected_gradient_vector_j + object_mij
                            ) * scaled_composite_jacobian
                        ).sum::<Scalar>()
                    ).sum::<Scalar>()
                ).collect()
            ).collect()
        ).sum())
    }
    fn calculate_nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalStiffnesses<N>, ConstitutiveError> {
        let deformation_gradients = self.calculate_deformation_gradients(nodal_coordinates);
        let normal_tangentss = Self::calculate_normal_tangents(nodal_coordinates);
        let objectss = self.calculate_objects(&Self::calculate_normal_gradients(nodal_coordinates));
        let mut scaled_traction = ZERO_VECTOR;
        Ok(self.get_constitutive_models().iter()
        .zip(deformation_gradients.iter())
        .map(|(constitutive_model, deformation_gradient)|
            constitutive_model.calculate_first_piola_kirchoff_stress(deformation_gradient)
        ).collect::<Result<FirstPiolaKirchoffStresses<G>, _>>()?.iter()
        .zip(self.get_constitutive_models().iter()
        .zip(deformation_gradients.iter())
        .map(|(constitutive_model, deformation_gradient)|
            constitutive_model.calculate_first_piola_kirchoff_tangent_stiffness(deformation_gradient)
        ).collect::<Result<FirstPiolaKirchoffTangentStiffnesses<G>, _>>()?.iter()
        .zip(self.get_projected_gradient_vectors().iter()
        .zip(self.get_integration_weights().iter()
        .zip(self.get_scaled_reference_normals().iter()
        .zip(objectss.iter())))))
        .map(|(first_piola_kirchoff_stress, (first_piola_kirchoff_tangent_stiffness, (projected_gradient_vectors, (scaled_composite_jacobian, (scaled_reference_normals, objects)))))|
            projected_gradient_vectors.iter()
            .zip(objects.iter())
            .map(|(projected_gradient_vector_a, object_a)|
                projected_gradient_vectors.iter()
                .zip(objects.iter())
                .map(|(projected_gradient_vector_b, object_b)|
                    IDENTITY.iter()
                    .zip(object_a.iter())
                    .map(|(identity_m, object_a_m)|
                        IDENTITY.iter()
                        .zip(object_b.iter())
                        .map(|(identity_n, object_b_n)|
                            first_piola_kirchoff_tangent_stiffness.iter()
                            .zip(identity_m.iter()
                            .zip(object_a_m.iter()))
                            .map(|(first_piola_kirchoff_tangent_stiffness_i, (identity_mi, object_a_mi))|
                                first_piola_kirchoff_tangent_stiffness_i.iter()
                                .zip(projected_gradient_vector_a.iter()
                                .zip(object_a_mi.iter()))
                                .map(|(first_piola_kirchoff_tangent_stiffness_ij, (projected_gradient_vector_a_j, object_a_mij))|
                                    first_piola_kirchoff_tangent_stiffness_ij.iter()
                                    .zip(identity_n.iter()
                                    .zip(object_b_n.iter()))
                                    .map(|(first_piola_kirchoff_tangent_stiffness_ijk, (identity_nk, object_b_nk))|
                                        first_piola_kirchoff_tangent_stiffness_ijk.iter()
                                        .zip(projected_gradient_vector_b.iter()
                                        .zip(object_b_nk.iter()))
                                        .map(|(first_piola_kirchoff_tangent_stiffness_ijkl, (projected_gradient_vector_b_l, object_b_nkl))|
                                            first_piola_kirchoff_tangent_stiffness_ijkl * (
                                                identity_mi * projected_gradient_vector_a_j + object_a_mij
                                            ) * (
                                                identity_nk * projected_gradient_vector_b_l + object_b_nkl
                                            ) * scaled_composite_jacobian
                                        ).sum::<Scalar>()
                                    ).sum::<Scalar>()
                                ).sum::<Scalar>()
                            ).sum::<Scalar>()
                        ).collect()
                    ).collect()
                ).collect()
            ).collect::<NodalStiffnesses<N>>() +
            normal_tangentss.iter()
            .zip(scaled_reference_normals.iter())
            .map(|(normal_tangents, scaled_reference_normal)|{
                scaled_traction = (first_piola_kirchoff_stress * scaled_reference_normal) * scaled_composite_jacobian;
                normal_tangents.iter()
                .map(|normal_tangent_a|
                    normal_tangent_a.iter()
                    .map(|normal_tangent_ab|
                        normal_tangent_ab.iter()
                        .map(|normal_tangent_ab_m|
                            normal_tangent_ab_m.iter()
                            .map(|normal_tangent_ab_mn|
                                normal_tangent_ab_mn * &scaled_traction
                            ).collect()
                        ).collect()
                    ).collect()
                ).collect::<NodalStiffnesses<N>>()
            }).sum::<NodalStiffnesses<N>>()
        ).sum())
    }
}

impl<'a, C> ViscoelasticFiniteElement<'a, C, G, N> for Triangle<C>
where
    C: Viscoelastic<'a>,
{
    fn calculate_nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalForces<N>, ConstitutiveError> {
        let normal_gradients = Self::calculate_normal_gradients(nodal_coordinates);
        Ok(self.get_constitutive_models().iter()
        .zip(self.calculate_deformation_gradients(nodal_coordinates).iter()
        .zip(self.calculate_deformation_gradient_rates(nodal_coordinates, nodal_velocities).iter()))
        .map(|(constitutive_model, (deformation_gradient, deformation_gradient_rate))|
            constitutive_model.calculate_first_piola_kirchoff_stress(deformation_gradient, deformation_gradient_rate)
        ).collect::<Result<FirstPiolaKirchoffStresses<G>, _>>()?.iter()
        .zip(self.get_projected_gradient_vectors().iter()
        .zip(self.get_integration_weights().iter()
        .zip(self.calculate_objects(&normal_gradients).iter())))
        .map(|(first_piola_kirchoff_stress, (projected_gradient_vectors, (scaled_composite_jacobian, objects)))|
            projected_gradient_vectors.iter()
            .zip(objects.iter())
            .map(|(projected_gradient_vector, object)|
                IDENTITY.iter()
                .zip(object.iter())
                .map(|(identity_m, object_m)|
                    first_piola_kirchoff_stress.iter()
                    .zip(identity_m.iter()
                    .zip(object_m.iter()))
                    .map(|(first_piola_kirchoff_stress_i, (identity_mi, object_mi))|
                        first_piola_kirchoff_stress_i.iter()
                        .zip(projected_gradient_vector.iter()
                        .zip(object_mi.iter()))
                        .map(|(first_piola_kirchoff_stress_ij, (projected_gradient_vector_j, object_mij))|
                            first_piola_kirchoff_stress_ij * (
                                identity_mi * projected_gradient_vector_j + object_mij
                            ) * scaled_composite_jacobian
                        ).sum::<Scalar>()
                    ).sum::<Scalar>()
                ).collect()
            ).collect()
        ).sum())
    }
    fn calculate_nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalStiffnesses<N>, ConstitutiveError> {
        let normal_gradients = Self::calculate_normal_gradients(nodal_coordinates);
        let objectss = self.calculate_objects(&normal_gradients);
        Ok(self.get_constitutive_models().iter()
        .zip(self.calculate_deformation_gradients(nodal_coordinates).iter()
        .zip(self.calculate_deformation_gradient_rates(nodal_coordinates, nodal_velocities).iter()))
        .map(|(constitutive_model, (deformation_gradient, deformation_gradient_rate))|
            constitutive_model.calculate_first_piola_kirchoff_rate_tangent_stiffness(deformation_gradient, deformation_gradient_rate)
        ).collect::<Result<FirstPiolaKirchoffRateTangentStiffnesses<G>, _>>()?.iter()
        .zip(self.get_projected_gradient_vectors().iter()
        .zip(self.get_integration_weights().iter()
        .zip(objectss.iter())))
        .map(|(first_piola_kirchoff_rate_tangent_stiffness, (projected_gradient_vectors, (scaled_composite_jacobian, objects)))|
            projected_gradient_vectors.iter()
            .zip(objects.iter())
            .map(|(projected_gradient_vector_a, object_a)|
                projected_gradient_vectors.iter()
                .zip(objects.iter())
                .map(|(projected_gradient_vector_b, object_b)|
                    IDENTITY.iter()
                    .zip(object_a.iter())
                    .map(|(identity_m, object_a_m)|
                        IDENTITY.iter()
                        .zip(object_b.iter())
                        .map(|(identity_n, object_b_n)|
                            first_piola_kirchoff_rate_tangent_stiffness.iter()
                            .zip(identity_m.iter()
                            .zip(object_a_m.iter()))
                            .map(|(first_piola_kirchoff_rate_tangent_stiffness_i, (identity_mi, object_a_mi))|
                                first_piola_kirchoff_rate_tangent_stiffness_i.iter()
                                .zip(projected_gradient_vector_a.iter()
                                .zip(object_a_mi.iter()))
                                .map(|(first_piola_kirchoff_rate_tangent_stiffness_ij, (projected_gradient_vector_a_j, object_a_mij))|
                                    first_piola_kirchoff_rate_tangent_stiffness_ij.iter()
                                    .zip(identity_n.iter()
                                    .zip(object_b_n.iter()))
                                    .map(|(first_piola_kirchoff_rate_tangent_stiffness_ijk, (identity_nk, object_b_nk))|
                                        first_piola_kirchoff_rate_tangent_stiffness_ijk.iter()
                                        .zip(projected_gradient_vector_b.iter()
                                        .zip(object_b_nk.iter()))
                                        .map(|(first_piola_kirchoff_rate_tangent_stiffness_ijkl, (projected_gradient_vector_b_l, object_b_nkl))|
                                            first_piola_kirchoff_rate_tangent_stiffness_ijkl * (
                                                identity_mi * projected_gradient_vector_a_j + object_a_mij
                                            ) * (
                                                identity_nk * projected_gradient_vector_b_l + object_b_nkl
                                            ) * scaled_composite_jacobian
                                        ).sum::<Scalar>()
                                    ).sum::<Scalar>()
                                ).sum::<Scalar>()
                            ).sum::<Scalar>()
                        ).collect()
                    ).collect()
                ).collect()
            ).collect()
        ).sum())
    }
}

impl<'a, C> CompositeSurfaceElement<'a, C, G, M, N, O, P, Q> for Triangle<C>
where
    C: Constitutive<'a>,
{
    fn get_scaled_reference_normals(&self) -> &ScaledReferenceNormals<G, P> {
        &self.scaled_reference_normals
    }
}

impl<'a, C> ElasticCompositeElement<'a, C, G, M, N, O, P, Q> for Triangle<C> where C: Elastic<'a> {}

impl<'a, C> HyperelasticFiniteElement<'a, C, G, N> for Triangle<C>
where
    C: Hyperelastic<'a>,
{
    fn calculate_helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.calculate_helmholtz_free_energy_composite_element(nodal_coordinates)
    }
}

impl<'a, C> HyperelasticCompositeElement<'a, C, G, M, N, O, P, Q> for Triangle<C> where
    C: Hyperelastic<'a>
{
}

impl<'a, C> ViscoelasticCompositeElement<'a, C, G, M, N, O, P, Q> for Triangle<C> where
    C: Viscoelastic<'a>
{
}

impl<'a, C> ElasticHyperviscousFiniteElement<'a, C, G, N> for Triangle<C>
where
    C: ElasticHyperviscous<'a>,
{
    fn calculate_viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.calculate_viscous_dissipation_composite_element(nodal_coordinates, nodal_velocities)
    }
    fn calculate_dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.calculate_dissipation_potential_composite_element(nodal_coordinates, nodal_velocities)
    }
}

impl<'a, C> ElasticHyperviscousCompositeElement<'a, C, G, M, N, O, P, Q> for Triangle<C> where
    C: ElasticHyperviscous<'a>
{
}

impl<'a, C> HyperviscoelasticFiniteElement<'a, C, G, N> for Triangle<C>
where
    C: Hyperviscoelastic<'a>,
{
    fn calculate_helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.calculate_helmholtz_free_energy_composite_element(nodal_coordinates)
    }
}

impl<'a, C> HyperviscoelasticCompositeElement<'a, C, G, M, N, O, P, Q> for Triangle<C> where
    C: Hyperviscoelastic<'a>
{
}
