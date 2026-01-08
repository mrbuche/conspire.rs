use crate::{
    fem::{
        NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::{
            Block,
            element::{
                ElementNodalCoordinates, ElementNodalReferenceCoordinates, ElementNodalVelocities,
                FiniteElement, GradientVectors,
                composite::tetrahedron::{G, M, N, P, Tetrahedron},
                quadratic::test::{
                    TETRAHEDRON_D as D, tetrahedron_applied_velocities as applied_velocities,
                    tetrahedron_applied_velocity as applied_velocity,
                    tetrahedron_equality_constraint as equality_constraint,
                    tetrahedron_get_connectivity as get_connectivity,
                    tetrahedron_get_coordinates_block as get_coordinates_block,
                    tetrahedron_get_reference_coordinates_block as get_reference_coordinates_block,
                    tetrahedron_get_velocities_block as get_velocities_block,
                    tetrahedron_reference_coordinates as reference_coordinates,
                },
                solid::SolidFiniteElement,
                test::test_finite_element,
            },
            test::test_finite_element_block,
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    math::{ScalarList, Tensor, TensorRank2, optimize::EqualityConstraint},
    mechanics::{DeformationGradient, DeformationGradientList, DeformationGradientRateList},
};

test_finite_element!(Tetrahedron);
test_finite_element_block!(Tetrahedron);

use crate::math::test::{TestError, assert_eq_within_tols};

#[test]
fn normalized_projection_matrix() -> Result<(), TestError> {
    Tetrahedron::shape_function_integrals_products()
        .iter()
        .map(|dummy| dummy * 1.0)
        .sum::<TensorRank2<P, 9, 9>>()
        .iter()
        .zip(
            Tetrahedron::inverse_normalized_projection_matrix()
                .inverse()
                .iter(),
        )
        .try_for_each(|(sum_i, projection_matrix_i)| {
            sum_i.iter().zip(projection_matrix_i.iter()).try_for_each(
                |(sum_ij, projection_matrix_ij)| {
                    assert_eq_within_tols(sum_ij, projection_matrix_ij)
                },
            )
        })
}

#[test]
fn standard_gradient_operators_transposed() -> Result<(), TestError> {
    let standard_gradient_operators_transposed =
        Tetrahedron::standard_gradient_operators_transposed();
    Tetrahedron::shape_functions_gradients_at_integration_points()
        .iter()
        .enumerate()
        .try_for_each(|(i, standard_gradient_operators_i)| {
            standard_gradient_operators_i
                .iter()
                .zip(standard_gradient_operators_transposed.iter())
                .try_for_each(
                    |(standard_gradient_operators_ij, standard_gradient_operators_transposed_j)| {
                        assert_eq_within_tols(
                            standard_gradient_operators_ij,
                            &standard_gradient_operators_transposed_j[i],
                        )
                    },
                )
        })
}
