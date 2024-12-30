#[cfg(test)]
pub mod test;

pub mod triangle;

use super::*;

pub trait CompositeSurfaceElement<
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
    Self: CompositeElement<'a, C, G, M, N, O, P, Q>,
{
    fn calculate_bases<const I: usize>(nodal_coordinates: &Coordinates<I, O>) -> Bases<I, P> {
        Self::calculate_standard_gradient_operators()
            .iter()
            .map(|standard_gradient_operator| {
                standard_gradient_operator
                    .iter()
                    .zip(nodal_coordinates.iter())
                    .map(|(standard_gradient_operator_a, nodal_coordinate_a)| {
                        standard_gradient_operator_a
                            .iter()
                            .map(|standard_gradient_operator_a_m| {
                                nodal_coordinate_a * standard_gradient_operator_a_m
                            })
                            .collect()
                    })
                    .sum()
            })
            .collect()
    }
    fn calculate_deformation_gradients_composite_surface_element(
        &self,
        nodal_coordinates: &NodalCoordinates<O>,
    ) -> DeformationGradients<G> {
        let normals = Self::calculate_normals(nodal_coordinates);
        self.get_projected_gradient_vectors()
            .iter()
            .zip(self.get_scaled_reference_normals().iter())
            .map(|(projected_gradient_vectors, scaled_reference_normals)| {
                nodal_coordinates
                    .iter()
                    .zip(projected_gradient_vectors.iter())
                    .map(|(nodal_coordinate, projected_gradient_vector)| {
                        DeformationGradient::dyad(nodal_coordinate, projected_gradient_vector)
                    })
                    .sum::<DeformationGradient>()
                    + scaled_reference_normals
                        .iter()
                        .zip(normals.iter())
                        .map(|(scaled_reference_normal, normal)| {
                            DeformationGradient::dyad(normal, scaled_reference_normal)
                        })
                        .sum::<DeformationGradient>()
            })
            .collect()
    }
    fn calculate_deformation_gradient_rates_composite_surface_element(
        &self,
        nodal_coordinates: &NodalCoordinates<O>,
        nodal_velocities: &NodalVelocities<O>,
    ) -> DeformationGradientRates<G> {
        let normal_rates = Self::calculate_normal_rates(nodal_coordinates, nodal_velocities);
        self.get_projected_gradient_vectors()
            .iter()
            .zip(self.get_scaled_reference_normals().iter())
            .map(|(projected_gradient_vectors, scaled_reference_normals)| {
                nodal_velocities
                    .iter()
                    .zip(projected_gradient_vectors.iter())
                    .map(|(nodal_velocity, projected_gradient_vector)| {
                        DeformationGradientRate::dyad(nodal_velocity, projected_gradient_vector)
                    })
                    .sum::<DeformationGradientRate>()
                    + scaled_reference_normals
                        .iter()
                        .zip(normal_rates.iter())
                        .map(|(scaled_reference_normal, normal_rate)| {
                            DeformationGradientRate::dyad(normal_rate, scaled_reference_normal)
                        })
                        .sum::<DeformationGradientRate>()
            })
            .collect()
    }
    fn calculate_dual_bases<const I: usize>(nodal_coordinates: &Coordinates<I, O>) -> Bases<I, P> {
        Self::calculate_bases(nodal_coordinates)
            .iter()
            .map(|basis_vectors| {
                basis_vectors
                    .iter()
                    .map(|basis_vectors_m| {
                        basis_vectors
                            .iter()
                            .map(|basis_vectors_n| basis_vectors_m * basis_vectors_n)
                            .collect()
                    })
                    .collect::<TensorRank2<M, I, I>>()
                    .inverse()
                    .iter()
                    .map(|metric_tensor_m| {
                        metric_tensor_m
                            .iter()
                            .zip(basis_vectors.iter())
                            .map(|(metric_tensor_mn, basis_vectors_n)| {
                                basis_vectors_n * metric_tensor_mn
                            })
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }
    fn calculate_normals(nodal_coordinates: &NodalCoordinates<O>) -> Normals<P> {
        Self::calculate_bases(nodal_coordinates)
            .iter()
            .map(|basis_vectors| basis_vectors[0].cross(&basis_vectors[1]).normalized())
            .collect()
    }
    fn calculate_normal_gradients(nodal_coordinates: &Coordinates<1, O>) -> NormalGradientss<P, O> {
        let levi_civita_symbol = LEVI_CIVITA;
        let mut normalization: Scalar = 0.0;
        let mut normal_vector = tensor_rank_1_zero();
        Self::calculate_standard_gradient_operators().iter()
        .zip(Self::calculate_bases(nodal_coordinates).iter())
        .map(|(standard_gradient_operator, basis_vectors)|{
            normalization = basis_vectors[0].cross(&basis_vectors[1]).norm();
            normal_vector = basis_vectors[0].cross(&basis_vectors[1])/normalization;
            standard_gradient_operator.iter()
            .map(|standard_gradient_operator_a|
                levi_civita_symbol.iter()
                .map(|levi_civita_symbol_m|
                    IDENTITY.iter()
                    .zip(normal_vector.iter())
                    .map(|(identity_i, normal_vector_i)|
                        levi_civita_symbol_m.iter()
                        .zip(basis_vectors[0].iter()
                        .zip(basis_vectors[1].iter()))
                        .map(|(levi_civita_symbol_mn, (basis_vector_0_n, basis_vector_1_n))|
                            levi_civita_symbol_mn.iter()
                            .zip(identity_i.iter()
                            .zip(normal_vector.iter()))
                            .map(|(levi_civita_symbol_mno, (identity_io, normal_vector_o))|
                                levi_civita_symbol_mno * (identity_io - normal_vector_i * normal_vector_o)
                            ).sum::<Scalar>() * (
                                standard_gradient_operator_a[0] * basis_vector_1_n
                              - standard_gradient_operator_a[1] * basis_vector_0_n
                            )
                        ).sum::<Scalar>() / normalization
                    ).collect()
                ).collect()
            ).collect()
        }).collect()
    }
    fn calculate_normal_rates(
        nodal_coordinates: &NodalCoordinates<O>,
        nodal_velocities: &NodalVelocities<O>,
    ) -> NormalRates<P> {
        let levi_civita_symbol = LEVI_CIVITA;
        let mut normalization: Scalar = 0.0;
        let mut normal_vector = tensor_rank_1_zero();
        Self::calculate_standard_gradient_operators().iter()
        .zip(Self::calculate_bases(nodal_coordinates).iter())
        .map(|(standard_gradient_operator, basis_vectors)|{
            normalization = basis_vectors[0].cross(&basis_vectors[1]).norm();
            normal_vector = basis_vectors[0].cross(&basis_vectors[1])/normalization;
            IDENTITY.iter()
            .zip(normal_vector.iter())
            .map(|(identity_i, normal_vector_i)|
                nodal_velocities.iter()
                .zip(standard_gradient_operator.iter())
                .map(|(nodal_velocity_a, standard_gradient_operator_a)|
                    levi_civita_symbol.iter()
                    .zip(nodal_velocity_a.iter())
                    .map(|(levi_civita_symbol_m, nodal_velocity_a_m)|
                        levi_civita_symbol_m.iter()
                        .zip(basis_vectors[0].iter()
                        .zip(basis_vectors[1].iter()))
                        .map(|(levi_civita_symbol_mn, (basis_vector_0_n, basis_vector_1_n))|
                            levi_civita_symbol_mn.iter()
                            .zip(identity_i.iter()
                            .zip(normal_vector.iter()))
                            .map(|(levi_civita_symbol_mno, (identity_io, normal_vector_o))|
                                levi_civita_symbol_mno * (identity_io - normal_vector_i * normal_vector_o)
                            ).sum::<Scalar>() * (
                                standard_gradient_operator_a[0] * basis_vector_1_n
                              - standard_gradient_operator_a[1] * basis_vector_0_n
                            )
                        ).sum::<Scalar>() * nodal_velocity_a_m
                    ).sum::<Scalar>()
                ).sum::<Scalar>() / normalization
            ).collect()
        }).collect()
    }
    fn calculate_normal_tangents(nodal_coordinates: &Coordinates<1, O>) -> NormalTangentss<P, O> {
        let levi_civita_symbol = LEVI_CIVITA;
        let mut normalization: Scalar = 0.0;
        let mut normal_vector = tensor_rank_1_zero();
        Self::calculate_standard_gradient_operators().iter()
        .zip(Self::calculate_bases(nodal_coordinates).iter()
        .zip(Self::calculate_normal_gradients(nodal_coordinates).iter()))
        .map(|(standard_gradient_operator, (basis_vectors, normal_gradients))|{
            normalization = basis_vectors[0].cross(&basis_vectors[1]).norm();
            normal_vector = basis_vectors[0].cross(&basis_vectors[1])/normalization;
            normal_gradients.iter()
            .zip(standard_gradient_operator.iter())
            .map(|(normal_gradient_a, standard_gradient_operator_a)|
                normal_gradients.iter()
                .zip(standard_gradient_operator.iter())
                .map(|(normal_gradient_b, standard_gradient_operator_b)|
                    normal_gradient_a.iter()
                    .zip(levi_civita_symbol.iter())
                    .map(|(normal_gradient_a_m, levi_civita_symbol_m)|
                        normal_gradient_b.iter()
                        .zip(levi_civita_symbol.iter()
                        .zip(levi_civita_symbol_m.iter()))
                        .map(|(normal_gradient_b_n, (levi_civita_symbol_n, levi_civita_symbol_mn))|
                            normal_gradient_a_m.iter()
                            .zip(normal_gradient_b_n.iter()
                            .zip(normal_vector.iter()
                            .zip(IDENTITY.iter())))
                            .map(|(normal_gradient_a_m_i, (normal_gradient_b_n_i, (normal_vector_i, identity_i)))|
                                (levi_civita_symbol_m.iter()
                                .zip(levi_civita_symbol_n.iter()
                                .zip(basis_vectors[0].iter()
                                .zip(basis_vectors[1].iter())))
                                .map(|(levi_civita_symbol_mr, (levi_civita_symbol_nr, (basis_vector_0_r, basis_vector_1_r)))|
                                    levi_civita_symbol_mr.iter()
                                    .zip(normal_vector.iter()
                                    .zip(normal_gradient_b_n.iter()))
                                    .map(|(levi_civita_symbol_mrs, (normal_vector_s, normal_gradient_b_n_s))|
                                        levi_civita_symbol_mrs * (
                                            normal_gradient_b_n_i * normal_vector_s
                                        + normal_gradient_b_n_s * normal_vector_i
                                        )
                                    ).sum::<Scalar>() * (
                                        standard_gradient_operator_a[1] * basis_vector_0_r
                                    - standard_gradient_operator_a[0] * basis_vector_1_r
                                    ) +
                                    levi_civita_symbol_nr.iter()
                                    .zip(normal_vector.iter())
                                    .map(|(levi_civita_symbol_nrs, normal_vector_s)|
                                        levi_civita_symbol_nrs * normal_vector_s * normal_gradient_a_m_i
                                    ).sum::<Scalar>() * (
                                        standard_gradient_operator_b[1] * basis_vector_0_r
                                    - standard_gradient_operator_b[0] * basis_vector_1_r
                                    )
                                ).sum::<Scalar>() +
                                levi_civita_symbol_mn * (
                                    identity_i - &normal_vector * normal_vector_i
                                ) * (
                                    standard_gradient_operator_a[0] * standard_gradient_operator_b[1]
                                - standard_gradient_operator_a[1] * standard_gradient_operator_b[0]
                                )) / normalization
                            ).collect()
                        ).collect()
                    ).collect()
                ).collect()
            ).collect()
        }).collect()
    }
    fn calculate_objects(
        &self,
        normal_gradients: &NormalGradientss<P, O>,
    ) -> TensorRank3List2D<3, 1, 1, 0, O, G> {
        self.get_scaled_reference_normals()
            .iter()
            .map(|scaled_reference_normals| {
                normal_gradients
                    .iter()
                    .zip(scaled_reference_normals.iter())
                    .map(|(normal_gradient, scaled_reference_normal)| {
                        normal_gradient
                            .iter()
                            .map(|normal_gradient_a| {
                                normal_gradient_a
                                    .iter()
                                    .map(|normal_gradient_a_m| {
                                        TensorRank2::dyad(
                                            normal_gradient_a_m,
                                            scaled_reference_normal,
                                        )
                                    })
                                    .collect::<TensorRank3<3, 1, 1, 0>>()
                            })
                            .collect::<TensorRank3List<3, 1, 1, 0, O>>()
                    })
                    .sum::<TensorRank3List<3, 1, 1, 0, O>>()
            })
            .collect()
    }
    fn calculate_projected_gradient_vectors_composite_surface_element(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
    ) -> ProjectedGradientVectors<G, N> {
        let reference_dual_bases = Self::calculate_dual_bases(reference_nodal_coordinates);
        let reference_jacobians_subelements =
            Self::calculate_reference_jacobians_subelements(reference_nodal_coordinates);
        let inverse_projection_matrix =
            Self::calculate_inverse_projection_matrix(&reference_jacobians_subelements);
        Self::calculate_shape_functions_at_integration_points()
            .iter()
            .map(|shape_functions_at_integration_point| {
                Self::calculate_standard_gradient_operators_transposed()
                    .iter()
                    .map(|standard_gradient_operators_a| {
                        Self::calculate_shape_function_integrals()
                            .iter()
                            .zip(
                                standard_gradient_operators_a.iter().zip(
                                    reference_dual_bases
                                        .iter()
                                        .zip(reference_jacobians_subelements.iter()),
                                ),
                            )
                            .map(
                                |(
                                    shape_function_integral,
                                    (
                                        standard_gradient_operator,
                                        (
                                            reference_dual_basis_vectors,
                                            reference_jacobian_subelement,
                                        ),
                                    ),
                                )| {
                                    reference_dual_basis_vectors
                                        .iter()
                                        .zip(standard_gradient_operator.iter())
                                        .map(
                                            |(
                                                reference_dual_basis_vector,
                                                standard_gradient_operator_mu,
                                            )| {
                                                reference_dual_basis_vector
                                                    * standard_gradient_operator_mu
                                            },
                                        )
                                        .sum::<Vector<0>>()
                                        * reference_jacobian_subelement
                                        * (shape_functions_at_integration_point
                                            * (&inverse_projection_matrix
                                                * shape_function_integral))
                                },
                            )
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }
    fn calculate_reference_normals(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
    ) -> ReferenceNormals<P> {
        Self::calculate_dual_bases(reference_nodal_coordinates)
            .iter()
            .map(|reference_dual_basis_vectors| {
                reference_dual_basis_vectors[0]
                    .cross(&reference_dual_basis_vectors[1])
                    .normalized()
            })
            .collect()
    }
    fn calculate_scaled_reference_normals(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
    ) -> ScaledReferenceNormals<G, P> {
        let reference_jacobians_subelements =
            Self::calculate_reference_jacobians_subelements(reference_nodal_coordinates);
        let inverse_projection_matrix =
            Self::calculate_inverse_projection_matrix(&reference_jacobians_subelements);
        let reference_normals = Self::calculate_reference_normals(reference_nodal_coordinates);
        let shape_function_integrals = Self::calculate_shape_function_integrals();
        Self::calculate_shape_functions_at_integration_points()
            .iter()
            .map(|shape_function| {
                reference_normals
                    .iter()
                    .zip(
                        shape_function_integrals
                            .iter()
                            .zip(reference_jacobians_subelements.iter()),
                    )
                    .map(
                        |(
                            reference_normal,
                            (shape_function_integral, reference_jacobian_subelement),
                        )| {
                            reference_normal
                                * ((shape_function
                                    * (&inverse_projection_matrix * shape_function_integral))
                                    * reference_jacobian_subelement)
                        },
                    )
                    .collect()
            })
            .collect()
    }
    fn get_scaled_reference_normals(&self) -> &ScaledReferenceNormals<G, P>;
}
