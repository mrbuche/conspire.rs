#[cfg(test)]
mod test;

use crate::{
    fem::{
        GradientVectors, NormalizedProjectionMatrix, ParametricGradientOperators, ProjectionMatrix,
        ShapeFunctionIntegrals, ShapeFunctionIntegralsProducts, ShapeFunctionsAtIntegrationPoints,
        StandardGradientOperators, StandardGradientOperatorsTransposed,
        block::element::{Element, ElementNodalReferenceCoordinates, FiniteElement},
    },
    math::{Scalar, Scalars, Tensor, TensorRank1},
};

const G: usize = 4;
const M: usize = 3;
const N: usize = 10;
const P: usize = 12;
const Q: usize = 4;

pub type Tetrahedron = Element<G, N>;

impl FiniteElement<G, N> for Tetrahedron {
    fn initialize(
        reference_nodal_coordinates: ElementNodalReferenceCoordinates<N>,
    ) -> (GradientVectors<G, N>, Scalars<G>) {
        let gradient_vectors = Self::projected_gradient_vectors(&reference_nodal_coordinates);
        let integration_weights =
            Self::reference_jacobians(&reference_nodal_coordinates) * Self::integration_weight();
        (gradient_vectors, integration_weights)
    }
    fn reset(&mut self) {
        let (gradient_vectors, integration_weights) = Self::initialize(Self::reference());
        self.gradient_vectors = gradient_vectors;
        self.integration_weights = integration_weights;
    }
}

impl Tetrahedron {
    const fn integration_weight() -> Scalar {
        1.0 / 24.0
    }
    const fn inverse_normalized_projection_matrix() -> NormalizedProjectionMatrix<Q> {
        const DIAG: Scalar = 4.0 / 640.0;
        const OFF: Scalar = -1.0 / 640.0;
        NormalizedProjectionMatrix::<Q>::const_from([
            [DIAG, OFF, OFF, OFF],
            [OFF, DIAG, OFF, OFF],
            [OFF, OFF, DIAG, OFF],
            [OFF, OFF, OFF, DIAG],
        ])
    }
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
        reference_nodal_coordinates: &ElementNodalReferenceCoordinates<N>,
    ) -> GradientVectors<G, N> {
        let parametric_gradient_operators = Self::standard_gradient_operators()
            .iter()
            .map(|standard_gradient_operator| {
                reference_nodal_coordinates * standard_gradient_operator
            })
            .collect::<ParametricGradientOperators<P>>();
        let reference_jacobians_subelements =
            Self::reference_jacobians_subelements(reference_nodal_coordinates);
        let inverse_projection_matrix =
            Self::inverse_projection_matrix(&reference_jacobians_subelements);
        Self::shape_functions_at_integration_points()
            .iter()
            .map(|shape_functions_at_integration_point| {
                Self::standard_gradient_operators_transposed()
                    .iter()
                    .map(|standard_gradient_operators_a| {
                        Self::shape_function_integrals()
                            .iter()
                            .zip(
                                standard_gradient_operators_a.iter().zip(
                                    parametric_gradient_operators
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
                                            parametric_gradient_operator,
                                            reference_jacobian_subelement,
                                        ),
                                    ),
                                )| {
                                    (parametric_gradient_operator.inverse_transpose()
                                        * standard_gradient_operator)
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
    const fn reference() -> ElementNodalReferenceCoordinates<N> {
        ElementNodalReferenceCoordinates::<N>::const_from([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.25, 0.25, 0.0],
            [0.25, 0.0, 0.25],
            [0.0, 0.25, 0.25],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ])
    }
    fn reference_jacobians(
        reference_nodal_coordinates: &ElementNodalReferenceCoordinates<N>,
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
        reference_nodal_coordinates: &ElementNodalReferenceCoordinates<N>,
    ) -> Scalars<P> {
        Self::standard_gradient_operators()
            .iter()
            .map(|standard_gradient_operator| {
                reference_nodal_coordinates * standard_gradient_operator
            })
            .collect::<ParametricGradientOperators<P>>()
            .iter()
            .map(|parametric_gradient_operator| parametric_gradient_operator.determinant())
            .collect()
    }
    const fn shape_functions_at_integration_points() -> ShapeFunctionsAtIntegrationPoints<G, Q> {
        const DIAG: Scalar = 0.585_410_196_624_968_5;
        const OFF: Scalar = 0.138_196_601_125_010_5;
        ShapeFunctionsAtIntegrationPoints::<G, Q>::const_from([
            [DIAG, OFF, OFF, OFF],
            [OFF, DIAG, OFF, OFF],
            [OFF, OFF, DIAG, OFF],
            [OFF, OFF, OFF, DIAG],
        ])
    }
    const fn shape_function_integrals() -> ShapeFunctionIntegrals<P, Q> {
        ShapeFunctionIntegrals::<P, Q>::const_from([
            [200.0, 40.0, 40.0, 40.0],
            [40.0, 200.0, 40.0, 40.0],
            [40.0, 40.0, 200.0, 40.0],
            [40.0, 40.0, 40.0, 200.0],
            [30.0, 70.0, 30.0, 30.0],
            [10.0, 50.0, 50.0, 50.0],
            [30.0, 30.0, 30.0, 70.0],
            [50.0, 50.0, 10.0, 50.0],
            [50.0, 50.0, 50.0, 10.0],
            [30.0, 30.0, 70.0, 30.0],
            [50.0, 10.0, 50.0, 50.0],
            [70.0, 30.0, 30.0, 30.0],
        ])
    }
    const fn shape_function_integrals_products() -> ShapeFunctionIntegralsProducts<P, Q> {
        ShapeFunctionIntegralsProducts::<P, Q>::const_from([
            [
                [128.0, 24.0, 24.0, 24.0],
                [24.0, 8.0, 4.0, 4.0],
                [24.0, 4.0, 8.0, 4.0],
                [24.0, 4.0, 4.0, 8.0],
            ],
            [
                [8.0, 24.0, 4.0, 4.0],
                [24.0, 128.0, 24.0, 24.0],
                [4.0, 24.0, 8.0, 4.0],
                [4.0, 24.0, 4.0, 8.0],
            ],
            [
                [8.0, 4.0, 24.0, 4.0],
                [4.0, 8.0, 24.0, 4.0],
                [24.0, 24.0, 128.0, 24.0],
                [4.0, 4.0, 24.0, 8.0],
            ],
            [
                [8.0, 4.0, 4.0, 24.0],
                [4.0, 8.0, 4.0, 24.0],
                [4.0, 4.0, 8.0, 24.0],
                [24.0, 24.0, 24.0, 128.0],
            ],
            [
                [7.0, 13.0, 5.0, 5.0],
                [13.0, 31.0, 13.0, 13.0],
                [5.0, 13.0, 7.0, 5.0],
                [5.0, 13.0, 5.0, 7.0],
            ],
            [
                [1.0, 3.0, 3.0, 3.0],
                [3.0, 17.0, 15.0, 15.0],
                [3.0, 15.0, 17.0, 15.0],
                [3.0, 15.0, 15.0, 17.0],
            ],
            [
                [7.0, 5.0, 5.0, 13.0],
                [5.0, 7.0, 5.0, 13.0],
                [5.0, 5.0, 7.0, 13.0],
                [13.0, 13.0, 13.0, 31.0],
            ],
            [
                [17.0, 15.0, 3.0, 15.0],
                [15.0, 17.0, 3.0, 15.0],
                [3.0, 3.0, 1.0, 3.0],
                [15.0, 15.0, 3.0, 17.0],
            ],
            [
                [17.0, 15.0, 15.0, 3.0],
                [15.0, 17.0, 15.0, 3.0],
                [15.0, 15.0, 17.0, 3.0],
                [3.0, 3.0, 3.0, 1.0],
            ],
            [
                [7.0, 5.0, 13.0, 5.0],
                [5.0, 7.0, 13.0, 5.0],
                [13.0, 13.0, 31.0, 13.0],
                [5.0, 5.0, 13.0, 7.0],
            ],
            [
                [17.0, 3.0, 15.0, 15.0],
                [3.0, 1.0, 3.0, 3.0],
                [15.0, 3.0, 17.0, 15.0],
                [15.0, 3.0, 15.0, 17.0],
            ],
            [
                [31.0, 13.0, 13.0, 13.0],
                [13.0, 7.0, 5.0, 5.0],
                [13.0, 5.0, 7.0, 5.0],
                [13.0, 5.0, 5.0, 7.0],
            ],
        ])
    }
    const fn standard_gradient_operators() -> StandardGradientOperators<M, N, P> {
        StandardGradientOperators::<M, N, P>::const_from([
            [
                [-2.0, -2.0, -2.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-2.0, -2.0, -2.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [-2.0, -2.0, -2.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-2.0, -2.0, -2.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-2.0 / 3.0, -2.0, -2.0],
                [4.0 / 3.0, 2.0, 0.0],
                [-2.0 / 3.0, 0.0, 0.0],
                [-2.0 / 3.0, 0.0, 0.0],
                [4.0 / 3.0, 0.0, 2.0],
                [-2.0 / 3.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0],
                [4.0 / 3.0, 4.0 / 3.0, -2.0 / 3.0],
                [-2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0],
                [-2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0],
                [4.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0],
                [-2.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -2.0 / 3.0],
                [0.0, 0.0, -2.0 / 3.0],
                [0.0, 0.0, -2.0 / 3.0],
                [-2.0, -2.0, -2.0 / 3.0],
                [2.0, 0.0, 4.0 / 3.0],
                [0.0, 2.0, 4.0 / 3.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, -4.0 / 3.0, -2.0],
                [0.0, 2.0 / 3.0, 0.0],
                [0.0, 2.0 / 3.0, 0.0],
                [-2.0, -4.0 / 3.0, 0.0],
                [2.0, 2.0 / 3.0, 2.0],
                [0.0, 2.0 / 3.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, -2.0, -4.0 / 3.0],
                [2.0, 2.0, 2.0 / 3.0],
                [-2.0, 0.0, -4.0 / 3.0],
                [0.0, 0.0, 2.0 / 3.0],
                [0.0, 0.0, 2.0 / 3.0],
                [0.0, 0.0, 2.0 / 3.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, -2.0 / 3.0, 0.0],
                [2.0, 4.0 / 3.0, 0.0],
                [-2.0, -2.0 / 3.0, -2.0],
                [0.0, -2.0 / 3.0, 0.0],
                [0.0, -2.0 / 3.0, 0.0],
                [0.0, 4.0 / 3.0, 2.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0 / 3.0, 0.0, 0.0],
                [2.0 / 3.0, 0.0, 0.0],
                [-4.0 / 3.0, 0.0, -2.0],
                [-4.0 / 3.0, -2.0, 0.0],
                [2.0 / 3.0, 0.0, 0.0],
                [2.0 / 3.0, 2.0, 2.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0 / 3.0, -4.0 / 3.0, -4.0 / 3.0],
                [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
                [-4.0 / 3.0, 2.0 / 3.0, -4.0 / 3.0],
                [-4.0 / 3.0, -4.0 / 3.0, 2.0 / 3.0],
                [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
                [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
            ],
        ])
    }
    const fn standard_gradient_operators_transposed() -> StandardGradientOperatorsTransposed<M, N, P>
    {
        StandardGradientOperatorsTransposed::<M, N, P>::const_from([
            [
                [-2.0, -2.0, -2.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [2.0, 0.0, 0.0],
                [-2.0, -2.0, -2.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-2.0 / 3.0, -2.0, -2.0],
                [-2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0],
                [0.0, 0.0, -2.0 / 3.0],
                [0.0, -4.0 / 3.0, -2.0],
                [0.0, -2.0, -4.0 / 3.0],
                [0.0, -2.0 / 3.0, 0.0],
                [2.0 / 3.0, 0.0, 0.0],
                [2.0 / 3.0, -4.0 / 3.0, -4.0 / 3.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [4.0 / 3.0, 2.0, 0.0],
                [4.0 / 3.0, 4.0 / 3.0, -2.0 / 3.0],
                [0.0, 0.0, -2.0 / 3.0],
                [0.0, 2.0 / 3.0, 0.0],
                [2.0, 2.0, 2.0 / 3.0],
                [2.0, 4.0 / 3.0, 0.0],
                [2.0 / 3.0, 0.0, 0.0],
                [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
            ],
            [
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
                [-2.0, -2.0, -2.0],
                [0.0, 0.0, 0.0],
                [-2.0 / 3.0, 0.0, 0.0],
                [-2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0],
                [0.0, 0.0, -2.0 / 3.0],
                [0.0, 2.0 / 3.0, 0.0],
                [-2.0, 0.0, -4.0 / 3.0],
                [-2.0, -2.0 / 3.0, -2.0],
                [-4.0 / 3.0, 0.0, -2.0],
                [-4.0 / 3.0, 2.0 / 3.0, -4.0 / 3.0],
            ],
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-2.0, -2.0, -2.0],
                [-2.0 / 3.0, 0.0, 0.0],
                [-2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0],
                [-2.0, -2.0, -2.0 / 3.0],
                [-2.0, -4.0 / 3.0, 0.0],
                [0.0, 0.0, 2.0 / 3.0],
                [0.0, -2.0 / 3.0, 0.0],
                [-4.0 / 3.0, -2.0, 0.0],
                [-4.0 / 3.0, -4.0 / 3.0, 2.0 / 3.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0 / 3.0, 0.0, 2.0],
                [4.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0],
                [2.0, 0.0, 4.0 / 3.0],
                [2.0, 2.0 / 3.0, 2.0],
                [0.0, 0.0, 2.0 / 3.0],
                [0.0, -2.0 / 3.0, 0.0],
                [2.0 / 3.0, 0.0, 0.0],
                [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 2.0, 0.0],
                [-2.0 / 3.0, 0.0, 0.0],
                [-2.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0],
                [0.0, 2.0, 4.0 / 3.0],
                [0.0, 2.0 / 3.0, 0.0],
                [0.0, 0.0, 2.0 / 3.0],
                [0.0, 4.0 / 3.0, 2.0],
                [2.0 / 3.0, 2.0, 2.0],
                [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
            ],
        ])
    }
}
