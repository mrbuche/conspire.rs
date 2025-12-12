#[cfg(test)]
mod test;

use crate::{
    fem::{
        GradientVectors, NormalizedProjectionMatrix, ParametricGradientOperators, ProjectionMatrix,
        ReferenceNodalCoordinates, ShapeFunctionIntegrals, ShapeFunctionIntegralsProducts,
        ShapeFunctionsAtIntegrationPoints, StandardGradientOperators,
        StandardGradientOperatorsTransposed,
        block::element::{Element, FiniteElement},
    },
    math::{
        Scalar, Scalars, Tensor, TensorRank1, TensorRank1List, TensorRank2, tensor_rank_1_zero,
    },
    mechanics::Coordinate,
};

const G: usize = 4;
const M: usize = 3;
const N: usize = 10;
const P: usize = 12;
const Q: usize = 4;

pub type Tetrahedron = Element<G, N>;

impl From<ReferenceNodalCoordinates<N>> for Tetrahedron {
    fn from(reference_nodal_coordinates: ReferenceNodalCoordinates<N>) -> Self {
        let (gradient_vectors, integration_weights) = Self::initialize(reference_nodal_coordinates);
        Self {
            gradient_vectors,
            integration_weights,
        }
    }
}

impl FiniteElement<G, N> for Tetrahedron {
    fn reference() -> ReferenceNodalCoordinates<N> {
        ReferenceNodalCoordinates::const_from([
            Coordinate::const_from([0.0, 0.0, 0.0]),
            Coordinate::const_from([1.0, 0.0, 0.0]),
            Coordinate::const_from([0.0, 1.0, 0.0]),
            Coordinate::const_from([0.0, 0.0, 1.0]),
            Coordinate::const_from([0.25, 0.25, 0.0]),
            Coordinate::const_from([0.25, 0.0, 0.25]),
            Coordinate::const_from([0.0, 0.25, 0.25]),
            Coordinate::const_from([0.5, 0.5, 0.0]),
            Coordinate::const_from([0.5, 0.0, 0.5]),
            Coordinate::const_from([0.0, 0.5, 0.5]),
        ])
    }
    fn reset(&mut self) {
        let (gradient_vectors, integration_weights) = Self::initialize(Self::reference());
        self.gradient_vectors = gradient_vectors;
        self.integration_weights = integration_weights;
    }
}

impl Tetrahedron {
    fn initialize(
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
    ) -> (GradientVectors<G, N>, Scalars<G>) {
        let gradient_vectors = Self::projected_gradient_vectors(&reference_nodal_coordinates);
        let integration_weights =
            Self::reference_jacobians(&reference_nodal_coordinates) * Self::integration_weight();
        (gradient_vectors, integration_weights)
    }
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
        reference_nodal_coordinates: &ReferenceNodalCoordinates<N>,
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
    fn reference_jacobians(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<N>,
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
        reference_nodal_coordinates: &ReferenceNodalCoordinates<N>,
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
        ShapeFunctionsAtIntegrationPoints::const_from([
            TensorRank1::const_from([DIAG, OFF, OFF, OFF]),
            TensorRank1::const_from([OFF, DIAG, OFF, OFF]),
            TensorRank1::const_from([OFF, OFF, DIAG, OFF]),
            TensorRank1::const_from([OFF, OFF, OFF, DIAG]),
        ])
    }
    const fn shape_function_integrals() -> ShapeFunctionIntegrals<P, Q> {
        ShapeFunctionIntegrals::const_from([
            TensorRank1::const_from([200.0, 40.0, 40.0, 40.0]),
            TensorRank1::const_from([40.0, 200.0, 40.0, 40.0]),
            TensorRank1::const_from([40.0, 40.0, 200.0, 40.0]),
            TensorRank1::const_from([40.0, 40.0, 40.0, 200.0]),
            TensorRank1::const_from([30.0, 70.0, 30.0, 30.0]),
            TensorRank1::const_from([10.0, 50.0, 50.0, 50.0]),
            TensorRank1::const_from([30.0, 30.0, 30.0, 70.0]),
            TensorRank1::const_from([50.0, 50.0, 10.0, 50.0]),
            TensorRank1::const_from([50.0, 50.0, 50.0, 10.0]),
            TensorRank1::const_from([30.0, 30.0, 70.0, 30.0]),
            TensorRank1::const_from([50.0, 10.0, 50.0, 50.0]),
            TensorRank1::const_from([70.0, 30.0, 30.0, 30.0]),
        ])
    }
    const fn shape_function_integrals_products() -> ShapeFunctionIntegralsProducts<P, Q> {
        ShapeFunctionIntegralsProducts::const_from([
            TensorRank2::<4, _, _>::const_from([
                [128.0, 24.0, 24.0, 24.0],
                [24.0, 8.0, 4.0, 4.0],
                [24.0, 4.0, 8.0, 4.0],
                [24.0, 4.0, 4.0, 8.0],
            ]),
            TensorRank2::<4, _, _>::const_from([
                [8.0, 24.0, 4.0, 4.0],
                [24.0, 128.0, 24.0, 24.0],
                [4.0, 24.0, 8.0, 4.0],
                [4.0, 24.0, 4.0, 8.0],
            ]),
            TensorRank2::<4, _, _>::const_from([
                [8.0, 4.0, 24.0, 4.0],
                [4.0, 8.0, 24.0, 4.0],
                [24.0, 24.0, 128.0, 24.0],
                [4.0, 4.0, 24.0, 8.0],
            ]),
            TensorRank2::<4, _, _>::const_from([
                [8.0, 4.0, 4.0, 24.0],
                [4.0, 8.0, 4.0, 24.0],
                [4.0, 4.0, 8.0, 24.0],
                [24.0, 24.0, 24.0, 128.0],
            ]),
            TensorRank2::<4, _, _>::const_from([
                [7.0, 13.0, 5.0, 5.0],
                [13.0, 31.0, 13.0, 13.0],
                [5.0, 13.0, 7.0, 5.0],
                [5.0, 13.0, 5.0, 7.0],
            ]),
            TensorRank2::<4, _, _>::const_from([
                [1.0, 3.0, 3.0, 3.0],
                [3.0, 17.0, 15.0, 15.0],
                [3.0, 15.0, 17.0, 15.0],
                [3.0, 15.0, 15.0, 17.0],
            ]),
            TensorRank2::<4, _, _>::const_from([
                [7.0, 5.0, 5.0, 13.0],
                [5.0, 7.0, 5.0, 13.0],
                [5.0, 5.0, 7.0, 13.0],
                [13.0, 13.0, 13.0, 31.0],
            ]),
            TensorRank2::<4, _, _>::const_from([
                [17.0, 15.0, 3.0, 15.0],
                [15.0, 17.0, 3.0, 15.0],
                [3.0, 3.0, 1.0, 3.0],
                [15.0, 15.0, 3.0, 17.0],
            ]),
            TensorRank2::<4, _, _>::const_from([
                [17.0, 15.0, 15.0, 3.0],
                [15.0, 17.0, 15.0, 3.0],
                [15.0, 15.0, 17.0, 3.0],
                [3.0, 3.0, 3.0, 1.0],
            ]),
            TensorRank2::<4, _, _>::const_from([
                [7.0, 5.0, 13.0, 5.0],
                [5.0, 7.0, 13.0, 5.0],
                [13.0, 13.0, 31.0, 13.0],
                [5.0, 5.0, 13.0, 7.0],
            ]),
            TensorRank2::<4, _, _>::const_from([
                [17.0, 3.0, 15.0, 15.0],
                [3.0, 1.0, 3.0, 3.0],
                [15.0, 3.0, 17.0, 15.0],
                [15.0, 3.0, 15.0, 17.0],
            ]),
            TensorRank2::<4, _, _>::const_from([
                [31.0, 13.0, 13.0, 13.0],
                [13.0, 7.0, 5.0, 5.0],
                [13.0, 5.0, 7.0, 5.0],
                [13.0, 5.0, 5.0, 7.0],
            ]),
        ])
    }
    const fn standard_gradient_operators() -> StandardGradientOperators<M, N, P> {
        const TWO_THIRDS: Scalar = 2.0 / 3.0;
        const FOUR_THIRDS: Scalar = 4.0 / 3.0;
        StandardGradientOperators::const_from([
            TensorRank1List::const_from([
                Coordinate::const_from([-2.0, -2.0, -2.0]),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([2.0, 0.0, 0.0]),
                tensor_rank_1_zero(),
                Coordinate::const_from([0.0, 2.0, 0.0]),
                Coordinate::const_from([0.0, 0.0, 2.0]),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
            ]),
            TensorRank1List::const_from([
                tensor_rank_1_zero(),
                Coordinate::const_from([2.0, 0.0, 0.0]),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([-2.0, -2.0, -2.0]),
                Coordinate::const_from([0.0, 2.0, 0.0]),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([0.0, 0.0, 2.0]),
                tensor_rank_1_zero(),
            ]),
            TensorRank1List::const_from([
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([0.0, 2.0, 0.0]),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([2.0, 0.0, 0.0]),
                Coordinate::const_from([-2.0, -2.0, -2.0]),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([0.0, 0.0, 2.0]),
            ]),
            TensorRank1List::const_from([
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([0.0, 0.0, 2.0]),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([-2.0, -2.0, -2.0]),
                Coordinate::const_from([2.0, 0.0, 0.0]),
                Coordinate::const_from([0.0, 2.0, 0.0]),
            ]),
            TensorRank1List::const_from([
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([-TWO_THIRDS, -2.0, -2.0]),
                Coordinate::const_from([FOUR_THIRDS, 2.0, 0.0]),
                Coordinate::const_from([-TWO_THIRDS, 0.0, 0.0]),
                Coordinate::const_from([-TWO_THIRDS, 0.0, 0.0]),
                Coordinate::const_from([FOUR_THIRDS, 0.0, 2.0]),
                Coordinate::const_from([-TWO_THIRDS, 0.0, 0.0]),
            ]),
            TensorRank1List::const_from([
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([-TWO_THIRDS, -TWO_THIRDS, -TWO_THIRDS]),
                Coordinate::const_from([FOUR_THIRDS, FOUR_THIRDS, -TWO_THIRDS]),
                Coordinate::const_from([-TWO_THIRDS, -TWO_THIRDS, -TWO_THIRDS]),
                Coordinate::const_from([-TWO_THIRDS, -TWO_THIRDS, -TWO_THIRDS]),
                Coordinate::const_from([FOUR_THIRDS, -TWO_THIRDS, FOUR_THIRDS]),
                Coordinate::const_from([-TWO_THIRDS, FOUR_THIRDS, FOUR_THIRDS]),
            ]),
            TensorRank1List::const_from([
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([0.0, 0.0, -TWO_THIRDS]),
                Coordinate::const_from([0.0, 0.0, -TWO_THIRDS]),
                Coordinate::const_from([0.0, 0.0, -TWO_THIRDS]),
                Coordinate::const_from([-2.0, -2.0, -TWO_THIRDS]),
                Coordinate::const_from([2.0, 0.0, FOUR_THIRDS]),
                Coordinate::const_from([0.0, 2.0, FOUR_THIRDS]),
            ]),
            TensorRank1List::const_from([
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([0.0, -FOUR_THIRDS, -2.0]),
                Coordinate::const_from([0.0, TWO_THIRDS, 0.0]),
                Coordinate::const_from([0.0, TWO_THIRDS, 0.0]),
                Coordinate::const_from([-2.0, -FOUR_THIRDS, 0.0]),
                Coordinate::const_from([2.0, TWO_THIRDS, 2.0]),
                Coordinate::const_from([0.0, TWO_THIRDS, 0.0]),
            ]),
            TensorRank1List::const_from([
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([0.0, -2.0, -FOUR_THIRDS]),
                Coordinate::const_from([2.0, 2.0, TWO_THIRDS]),
                Coordinate::const_from([-2.0, 0.0, -FOUR_THIRDS]),
                Coordinate::const_from([0.0, 0.0, TWO_THIRDS]),
                Coordinate::const_from([0.0, 0.0, TWO_THIRDS]),
                Coordinate::const_from([0.0, 0.0, TWO_THIRDS]),
            ]),
            TensorRank1List::const_from([
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([0.0, -TWO_THIRDS, 0.0]),
                Coordinate::const_from([2.0, FOUR_THIRDS, 0.0]),
                Coordinate::const_from([-2.0, -TWO_THIRDS, -2.0]),
                Coordinate::const_from([0.0, -TWO_THIRDS, 0.0]),
                Coordinate::const_from([0.0, -TWO_THIRDS, 0.0]),
                Coordinate::const_from([0.0, FOUR_THIRDS, 2.0]),
            ]),
            TensorRank1List::const_from([
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([TWO_THIRDS, 0.0, 0.0]),
                Coordinate::const_from([TWO_THIRDS, 0.0, 0.0]),
                Coordinate::const_from([-FOUR_THIRDS, 0.0, -2.0]),
                Coordinate::const_from([-FOUR_THIRDS, -2.0, 0.0]),
                Coordinate::const_from([TWO_THIRDS, 0.0, 0.0]),
                Coordinate::const_from([TWO_THIRDS, 2.0, 2.0]),
            ]),
            TensorRank1List::const_from([
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([TWO_THIRDS, -FOUR_THIRDS, -FOUR_THIRDS]),
                Coordinate::const_from([TWO_THIRDS, TWO_THIRDS, TWO_THIRDS]),
                Coordinate::const_from([-FOUR_THIRDS, TWO_THIRDS, -FOUR_THIRDS]),
                Coordinate::const_from([-FOUR_THIRDS, -FOUR_THIRDS, TWO_THIRDS]),
                Coordinate::const_from([TWO_THIRDS, TWO_THIRDS, TWO_THIRDS]),
                Coordinate::const_from([TWO_THIRDS, TWO_THIRDS, TWO_THIRDS]),
            ]),
        ])
    }
    const fn standard_gradient_operators_transposed() -> StandardGradientOperatorsTransposed<M, N, P>
    {
        const TWO_THIRDS: Scalar = 2.0 / 3.0;
        const FOUR_THIRDS: Scalar = 4.0 / 3.0;
        StandardGradientOperators::const_from([
            TensorRank1List::const_from([
                Coordinate::const_from([-2.0, -2.0, -2.0]),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
            ]),
            TensorRank1List::const_from([
                tensor_rank_1_zero(),
                Coordinate::const_from([2.0, 0.0, 0.0]),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
            ]),
            TensorRank1List::const_from([
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([0.0, 2.0, 0.0]),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
            ]),
            TensorRank1List::const_from([
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([0.0, 0.0, 2.0]),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
            ]),
            TensorRank1List::const_from([
                Coordinate::const_from([2.0, 0.0, 0.0]),
                Coordinate::const_from([-2.0, -2.0, -2.0]),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([-TWO_THIRDS, -2.0, -2.0]),
                Coordinate::const_from([-TWO_THIRDS, -TWO_THIRDS, -TWO_THIRDS]),
                Coordinate::const_from([0.0, 0.0, -TWO_THIRDS]),
                Coordinate::const_from([0.0, -FOUR_THIRDS, -2.0]),
                Coordinate::const_from([0.0, -2.0, -FOUR_THIRDS]),
                Coordinate::const_from([0.0, -TWO_THIRDS, 0.0]),
                Coordinate::const_from([TWO_THIRDS, 0.0, 0.0]),
                Coordinate::const_from([TWO_THIRDS, -FOUR_THIRDS, -FOUR_THIRDS]),
            ]),
            TensorRank1List::const_from([
                tensor_rank_1_zero(),
                Coordinate::const_from([0.0, 2.0, 0.0]),
                Coordinate::const_from([2.0, 0.0, 0.0]),
                tensor_rank_1_zero(),
                Coordinate::const_from([FOUR_THIRDS, 2.0, 0.0]),
                Coordinate::const_from([FOUR_THIRDS, FOUR_THIRDS, -TWO_THIRDS]),
                Coordinate::const_from([0.0, 0.0, -TWO_THIRDS]),
                Coordinate::const_from([0.0, TWO_THIRDS, 0.0]),
                Coordinate::const_from([2.0, 2.0, TWO_THIRDS]),
                Coordinate::const_from([2.0, FOUR_THIRDS, 0.0]),
                Coordinate::const_from([TWO_THIRDS, 0.0, 0.0]),
                Coordinate::const_from([TWO_THIRDS, TWO_THIRDS, TWO_THIRDS]),
            ]),
            TensorRank1List::const_from([
                Coordinate::const_from([0.0, 2.0, 0.0]),
                tensor_rank_1_zero(),
                Coordinate::const_from([-2.0, -2.0, -2.0]),
                tensor_rank_1_zero(),
                Coordinate::const_from([-TWO_THIRDS, 0.0, 0.0]),
                Coordinate::const_from([-TWO_THIRDS, -TWO_THIRDS, -TWO_THIRDS]),
                Coordinate::const_from([0.0, 0.0, -TWO_THIRDS]),
                Coordinate::const_from([0.0, TWO_THIRDS, 0.0]),
                Coordinate::const_from([-2.0, 0.0, -FOUR_THIRDS]),
                Coordinate::const_from([-2.0, -TWO_THIRDS, -2.0]),
                Coordinate::const_from([-FOUR_THIRDS, 0.0, -2.0]),
                Coordinate::const_from([-FOUR_THIRDS, TWO_THIRDS, -FOUR_THIRDS]),
            ]),
            TensorRank1List::const_from([
                Coordinate::const_from([0.0, 0.0, 2.0]),
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([-2.0, -2.0, -2.0]),
                Coordinate::const_from([-TWO_THIRDS, 0.0, 0.0]),
                Coordinate::const_from([-TWO_THIRDS, -TWO_THIRDS, -TWO_THIRDS]),
                Coordinate::const_from([-2.0, -2.0, -TWO_THIRDS]),
                Coordinate::const_from([-2.0, -FOUR_THIRDS, 0.0]),
                Coordinate::const_from([0.0, 0.0, TWO_THIRDS]),
                Coordinate::const_from([0.0, -TWO_THIRDS, 0.0]),
                Coordinate::const_from([-FOUR_THIRDS, -2.0, 0.0]),
                Coordinate::const_from([-FOUR_THIRDS, -FOUR_THIRDS, TWO_THIRDS]),
            ]),
            TensorRank1List::const_from([
                tensor_rank_1_zero(),
                Coordinate::const_from([0.0, 0.0, 2.0]),
                tensor_rank_1_zero(),
                Coordinate::const_from([2.0, 0.0, 0.0]),
                Coordinate::const_from([FOUR_THIRDS, 0.0, 2.0]),
                Coordinate::const_from([FOUR_THIRDS, -TWO_THIRDS, FOUR_THIRDS]),
                Coordinate::const_from([2.0, 0.0, FOUR_THIRDS]),
                Coordinate::const_from([2.0, TWO_THIRDS, 2.0]),
                Coordinate::const_from([0.0, 0.0, TWO_THIRDS]),
                Coordinate::const_from([0.0, -TWO_THIRDS, 0.0]),
                Coordinate::const_from([TWO_THIRDS, 0.0, 0.0]),
                Coordinate::const_from([TWO_THIRDS, TWO_THIRDS, TWO_THIRDS]),
            ]),
            TensorRank1List::const_from([
                tensor_rank_1_zero(),
                tensor_rank_1_zero(),
                Coordinate::const_from([0.0, 0.0, 2.0]),
                Coordinate::const_from([0.0, 2.0, 0.0]),
                Coordinate::const_from([-TWO_THIRDS, 0.0, 0.0]),
                Coordinate::const_from([-TWO_THIRDS, FOUR_THIRDS, FOUR_THIRDS]),
                Coordinate::const_from([0.0, 2.0, FOUR_THIRDS]),
                Coordinate::const_from([0.0, TWO_THIRDS, 0.0]),
                Coordinate::const_from([0.0, 0.0, TWO_THIRDS]),
                Coordinate::const_from([0.0, FOUR_THIRDS, 2.0]),
                Coordinate::const_from([TWO_THIRDS, 2.0, 2.0]),
                Coordinate::const_from([TWO_THIRDS, TWO_THIRDS, TWO_THIRDS]),
            ]),
        ])
    }
}
