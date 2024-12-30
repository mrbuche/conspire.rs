#[cfg(test)]
pub mod test;

pub mod triangle;

use super::*;

pub trait LinearSurfaceElement<
    'a,
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const O: usize,
> where
    C: Constitutive<'a>,
    Self: LinearElement<'a, C, G, M, N, O>,
{
    fn calculate_basis<const I: usize>(nodal_coordinates: &Coordinates<I, O>) -> Basis<I> {
        Self::calculate_standard_gradient_operator()
            .iter()
            .zip(nodal_coordinates.iter())
            .map(|(standard_gradient_operator_a, nodal_coordinates_a)| {
                standard_gradient_operator_a
                    .iter()
                    .map(|standard_gradient_operator_a_m| {
                        nodal_coordinates_a * standard_gradient_operator_a_m
                    })
                    .collect()
            })
            .sum()
    }
    fn calculate_deformation_gradient_linear_surface_element(
        &self,
        nodal_coordinates: &NodalCoordinates<O>,
    ) -> DeformationGradient {
        nodal_coordinates
            .iter()
            .zip(self.get_gradient_vectors().iter())
            .map(|(nodal_coordinate, gradient_vector)| {
                DeformationGradient::dyad(nodal_coordinate, gradient_vector)
            })
            .sum::<DeformationGradient>()
            + DeformationGradient::dyad(
                &Self::calculate_normal(nodal_coordinates),
                self.get_reference_normal(),
            )
    }
    fn calculate_deformation_gradient_rate_linear_surface_element(
        &self,
        nodal_coordinates: &NodalCoordinates<O>,
        nodal_velocities: &NodalVelocities<O>,
    ) -> DeformationGradientRate {
        nodal_velocities
            .iter()
            .zip(self.get_gradient_vectors().iter())
            .map(|(nodal_velocity, gradient_vector)| {
                DeformationGradientRate::dyad(nodal_velocity, gradient_vector)
            })
            .sum::<DeformationGradientRate>()
            + DeformationGradientRate::dyad(
                &Self::calculate_normal_rate(nodal_coordinates, nodal_velocities),
                self.get_reference_normal(),
            )
    }
    fn calculate_dual_basis<const I: usize>(nodal_coordinates: &Coordinates<I, O>) -> Basis<I> {
        let basis_vectors = Self::calculate_basis(nodal_coordinates);
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
                    .map(|(metric_tensor_mn, basis_vectors_n)| basis_vectors_n * metric_tensor_mn)
                    .sum()
            })
            .collect()
    }
    fn calculate_gradient_vectors_linear_surface_element(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
    ) -> GradientVectors<N> {
        let reference_dual_basis_vectors = Self::calculate_dual_basis(reference_nodal_coordinates);
        Self::calculate_standard_gradient_operator()
            .iter()
            .map(|standard_gradient_operator_a| {
                standard_gradient_operator_a
                    .iter()
                    .zip(reference_dual_basis_vectors.iter())
                    .map(
                        |(standard_gradient_operator_a_m, dual_reference_basis_vector_m)| {
                            dual_reference_basis_vector_m * standard_gradient_operator_a_m
                        },
                    )
                    .sum()
            })
            .collect()
    }
    fn calculate_normal(nodal_coordinates: &NodalCoordinates<O>) -> Normal {
        let basis_vectors = Self::calculate_basis(nodal_coordinates);
        basis_vectors[0].cross(&basis_vectors[1]).normalized()
    }
    fn calculate_normal_gradients(nodal_coordinates: &Coordinates<1, O>) -> NormalGradients<O> {
        let basis_vectors = Self::calculate_basis(nodal_coordinates);
        let levi_civita_symbol = LEVI_CIVITA;
        let normalization = basis_vectors[0].cross(&basis_vectors[1]).norm();
        let normal_vector = basis_vectors[0].cross(&basis_vectors[1]) / normalization;
        Self::calculate_standard_gradient_operator().iter()
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
    }
    fn calculate_normal_rate(
        nodal_coordinates: &NodalCoordinates<O>,
        nodal_velocities: &NodalVelocities<O>,
    ) -> NormalRate {
        let basis_vectors = Self::calculate_basis(nodal_coordinates);
        let levi_civita_symbol = LEVI_CIVITA;
        let normalization = basis_vectors[0].cross(&basis_vectors[1]).norm();
        let normal_vector = basis_vectors[0].cross(&basis_vectors[1]) / normalization;
        let standard_gradient_operator = Self::calculate_standard_gradient_operator();
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
    }
    fn calculate_normal_tangents(nodal_coordinates: &Coordinates<1, O>) -> NormalTangents<O> {
        let basis_vectors = Self::calculate_basis(nodal_coordinates);
        let levi_civita_symbol = LEVI_CIVITA;
        let normalization = basis_vectors[0].cross(&basis_vectors[1]).norm();
        let normal_gradients = Self::calculate_normal_gradients(nodal_coordinates);
        let normal_vector = basis_vectors[0].cross(&basis_vectors[1]) / normalization;
        let standard_gradient_operator = Self::calculate_standard_gradient_operator();
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
    }
    fn calculate_reference_jacobian_linear_surface_element(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
    ) -> Scalar {
        let reference_basis_vectors = Self::calculate_basis(reference_nodal_coordinates);
        reference_basis_vectors[0]
            .cross(&reference_basis_vectors[1])
            .norm()
    }
    fn calculate_reference_normal(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
    ) -> ReferenceNormal {
        let dual_basis_vectors = Self::calculate_dual_basis(reference_nodal_coordinates);
        dual_basis_vectors[0]
            .cross(&dual_basis_vectors[1])
            .normalized()
    }
    fn get_reference_normal(&self) -> &ReferenceNormal;
}
