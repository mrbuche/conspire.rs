#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        ElementNodalReferenceCoordinates, FiniteElement, GradientVectors, ParametricCoordinate,
        ParametricCoordinates, ParametricReference, ShapeFunctions,
        ShapeFunctionsAtIntegrationPoints, ShapeFunctionsGradients, StandardGradientOperators,
        StandardGradientOperatorsTransposed,
        composite::{
            CompositeElement, NormalizedProjectionMatrix, ParametricGradientOperators,
            ProjectionMatrix, ShapeFunctionIntegrals, ShapeFunctionIntegralsProducts,
        },
        quadratic::Tetrahedron as QuadraticTetrahedron,
    },
    math::{Scalar, ScalarList, Tensor, TensorRank1},
};

const G: usize = 4;
const M: usize = 3;
const N: usize = 10;
const P: usize = 4;
const Q: usize = 12;

pub type Tetrahedron = CompositeElement<G, N>;

impl From<ElementNodalReferenceCoordinates<N>> for Tetrahedron {
    fn from(reference_nodal_coordinates: ElementNodalReferenceCoordinates<N>) -> Self {
        let gradient_vectors = Self::projected_gradient_vectors(&reference_nodal_coordinates);
        let integration_weights =
            Self::reference_jacobians(&reference_nodal_coordinates) * Self::integration_weight();
        Self {
            gradient_vectors,
            integration_weights,
        }
    }
}

impl FiniteElement<G, M, N, P> for Tetrahedron {
    fn integration_points() -> ParametricCoordinates<G, M> {
        QuadraticTetrahedron::integration_points() // temporary
    }
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
        .into()
    }
    fn parametric_weights() -> ScalarList<G> {
        [1.0 / 24.0; G].into()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<P> {
        todo!()
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, P> {
        todo!()
    }
}

impl Tetrahedron {
    const fn integration_weight() -> Scalar {
        1.0 / 24.0
    }
    fn inverse_normalized_projection_matrix() -> NormalizedProjectionMatrix<P> {
        const DIAG: Scalar = 4.0 / 640.0;
        const OFF: Scalar = -1.0 / 640.0;
        [
            [DIAG, OFF, OFF, OFF],
            [OFF, DIAG, OFF, OFF],
            [OFF, OFF, DIAG, OFF],
            [OFF, OFF, OFF, DIAG],
        ]
        .into()
    }
    fn inverse_projection_matrix(
        reference_jacobians_subelements: &ScalarList<Q>,
    ) -> NormalizedProjectionMatrix<P> {
        Self::shape_function_integrals_products()
            .iter()
            .zip(reference_jacobians_subelements)
            .map(
                |(shape_function_integrals_products, reference_jacobian_subelement)| {
                    shape_function_integrals_products * reference_jacobian_subelement
                },
            )
            .sum::<ProjectionMatrix<P>>()
            .inverse()
    }
    fn projected_gradient_vectors(
        reference_nodal_coordinates: &ElementNodalReferenceCoordinates<N>,
    ) -> GradientVectors<G, N> {
        let parametric_gradient_operators = Self::shape_functions_gradients_at_integration_points()
            .iter()
            .map(|standard_gradient_operator| {
                reference_nodal_coordinates * standard_gradient_operator
            })
            .collect::<ParametricGradientOperators<Q>>();
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
    fn reference_jacobians(
        reference_nodal_coordinates: &ElementNodalReferenceCoordinates<N>,
    ) -> ScalarList<G> {
        let vector = Self::inverse_normalized_projection_matrix()
            * Self::shape_function_integrals()
                .iter()
                .zip(Self::reference_jacobians_subelements(
                    reference_nodal_coordinates,
                ))
                .map(|(shape_function_integral, reference_jacobian_subelement)| {
                    shape_function_integral * reference_jacobian_subelement
                })
                .sum::<TensorRank1<P, 9>>();
        Self::shape_functions_at_integration_points()
            .iter()
            .map(|shape_functions_at_integration_point| {
                shape_functions_at_integration_point * &vector
            })
            .collect()
    }
    fn reference_jacobians_subelements(
        reference_nodal_coordinates: &ElementNodalReferenceCoordinates<N>,
    ) -> ScalarList<Q> {
        Self::shape_functions_gradients_at_integration_points()
            .iter()
            .map(|standard_gradient_operator| {
                reference_nodal_coordinates * standard_gradient_operator
            })
            .collect::<ParametricGradientOperators<Q>>()
            .iter()
            .map(|parametric_gradient_operator| parametric_gradient_operator.determinant())
            .collect()
    }
    fn shape_functions_at_integration_points() -> ShapeFunctionsAtIntegrationPoints<G, P> {
        const DIAG: Scalar = 0.585_410_196_624_968_5;
        const OFF: Scalar = 0.138_196_601_125_010_5;
        [
            [DIAG, OFF, OFF, OFF],
            [OFF, DIAG, OFF, OFF],
            [OFF, OFF, DIAG, OFF],
            [OFF, OFF, OFF, DIAG],
        ]
        .into()
    }
    fn shape_function_integrals() -> ShapeFunctionIntegrals<Q, P> {
        [
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
        ]
        .into()
    }
    fn shape_function_integrals_products() -> ShapeFunctionIntegralsProducts<Q, P> {
        [
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
        ]
        .into()
    }
    fn shape_functions_gradients_at_integration_points() -> StandardGradientOperators<M, N, Q> {
        [
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
        ]
        .into()
    }
    fn standard_gradient_operators_transposed() -> StandardGradientOperatorsTransposed<M, N, Q> {
        [
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
        ]
        .into()
    }
}
