#[cfg(test)]
mod test;

use super::*;
use crate::math::{FOUR_THIRDS, TWO_THIRDS};

const G: usize = 4;
const M: usize = 3;
const N: usize = 10;
const O: usize = 10;
const P: usize = 12;
const Q: usize = 4;

const INTEGRATION_WEIGHT: Scalar = ONE_TWENTY_FOURTH;

const INVERSE_NORMALIED_PROJECTION_MATRIX: NormalizedProjectionMatrix<Q> =
    TensorRank2::<Q, 9, 9>([
        TensorRank1([4.0 / 640.0, -1.0 / 640.0, -1.0 / 640.0, -1.0 / 640.0]),
        TensorRank1([-1.0 / 640.0, 4.0 / 640.0, -1.0 / 640.0, -1.0 / 640.0]),
        TensorRank1([-1.0 / 640.0, -1.0 / 640.0, 4.0 / 640.0, -1.0 / 640.0]),
        TensorRank1([-1.0 / 640.0, -1.0 / 640.0, -1.0 / 640.0, 4.0 / 640.0]),
    ]);
const SHAPE_FUNCTION_INTEGRALS: ShapeFunctionIntegrals<P, Q> = TensorRank1List::<Q, 9, P>([
    TensorRank1([200.0, 40.0, 40.0, 40.0]),
    TensorRank1([40.0, 200.0, 40.0, 40.0]),
    TensorRank1([40.0, 40.0, 200.0, 40.0]),
    TensorRank1([40.0, 40.0, 40.0, 200.0]),
    TensorRank1([30.0, 70.0, 30.0, 30.0]),
    TensorRank1([10.0, 50.0, 50.0, 50.0]),
    TensorRank1([30.0, 30.0, 30.0, 70.0]),
    TensorRank1([50.0, 50.0, 10.0, 50.0]),
    TensorRank1([50.0, 50.0, 50.0, 10.0]),
    TensorRank1([30.0, 30.0, 70.0, 30.0]),
    TensorRank1([50.0, 10.0, 50.0, 50.0]),
    TensorRank1([70.0, 30.0, 30.0, 30.0]),
]);
const SHAPE_FUNCTION_INTEGRALS_PRODUCTS: ShapeFunctionIntegralsProducts<P, Q> =
    TensorRank2List::<Q, 9, 9, P>([
        TensorRank2([
            TensorRank1([128.0, 24.0, 24.0, 24.0]),
            TensorRank1([24.0, 8.0, 4.0, 4.0]),
            TensorRank1([24.0, 4.0, 8.0, 4.0]),
            TensorRank1([24.0, 4.0, 4.0, 8.0]),
        ]),
        TensorRank2([
            TensorRank1([8.0, 24.0, 4.0, 4.0]),
            TensorRank1([24.0, 128.0, 24.0, 24.0]),
            TensorRank1([4.0, 24.0, 8.0, 4.0]),
            TensorRank1([4.0, 24.0, 4.0, 8.0]),
        ]),
        TensorRank2([
            TensorRank1([8.0, 4.0, 24.0, 4.0]),
            TensorRank1([4.0, 8.0, 24.0, 4.0]),
            TensorRank1([24.0, 24.0, 128.0, 24.0]),
            TensorRank1([4.0, 4.0, 24.0, 8.0]),
        ]),
        TensorRank2([
            TensorRank1([8.0, 4.0, 4.0, 24.0]),
            TensorRank1([4.0, 8.0, 4.0, 24.0]),
            TensorRank1([4.0, 4.0, 8.0, 24.0]),
            TensorRank1([24.0, 24.0, 24.0, 128.0]),
        ]),
        TensorRank2([
            TensorRank1([7.0, 13.0, 5.0, 5.0]),
            TensorRank1([13.0, 31.0, 13.0, 13.0]),
            TensorRank1([5.0, 13.0, 7.0, 5.0]),
            TensorRank1([5.0, 13.0, 5.0, 7.0]),
        ]),
        TensorRank2([
            TensorRank1([1.0, 3.0, 3.0, 3.0]),
            TensorRank1([3.0, 17.0, 15.0, 15.0]),
            TensorRank1([3.0, 15.0, 17.0, 15.0]),
            TensorRank1([3.0, 15.0, 15.0, 17.0]),
        ]),
        TensorRank2([
            TensorRank1([7.0, 5.0, 5.0, 13.0]),
            TensorRank1([5.0, 7.0, 5.0, 13.0]),
            TensorRank1([5.0, 5.0, 7.0, 13.0]),
            TensorRank1([13.0, 13.0, 13.0, 31.0]),
        ]),
        TensorRank2([
            TensorRank1([17.0, 15.0, 3.0, 15.0]),
            TensorRank1([15.0, 17.0, 3.0, 15.0]),
            TensorRank1([3.0, 3.0, 1.0, 3.0]),
            TensorRank1([15.0, 15.0, 3.0, 17.0]),
        ]),
        TensorRank2([
            TensorRank1([17.0, 15.0, 15.0, 3.0]),
            TensorRank1([15.0, 17.0, 15.0, 3.0]),
            TensorRank1([15.0, 15.0, 17.0, 3.0]),
            TensorRank1([3.0, 3.0, 3.0, 1.0]),
        ]),
        TensorRank2([
            TensorRank1([7.0, 5.0, 13.0, 5.0]),
            TensorRank1([5.0, 7.0, 13.0, 5.0]),
            TensorRank1([13.0, 13.0, 31.0, 13.0]),
            TensorRank1([5.0, 5.0, 13.0, 7.0]),
        ]),
        TensorRank2([
            TensorRank1([17.0, 3.0, 15.0, 15.0]),
            TensorRank1([3.0, 1.0, 3.0, 3.0]),
            TensorRank1([15.0, 3.0, 17.0, 15.0]),
            TensorRank1([15.0, 3.0, 15.0, 17.0]),
        ]),
        TensorRank2([
            TensorRank1([31.0, 13.0, 13.0, 13.0]),
            TensorRank1([13.0, 7.0, 5.0, 5.0]),
            TensorRank1([13.0, 5.0, 7.0, 5.0]),
            TensorRank1([13.0, 5.0, 5.0, 7.0]),
        ]),
    ]);
const DIAG: Scalar = 0.585_410_196_624_968_5;
const OFF: Scalar = 0.138_196_601_125_010_5;
const SHAPE_FUNCTIONS_AT_INTEGRATION_POINTS: ShapeFunctionsAtIntegrationPoints<G, Q> =
    TensorRank1List::<Q, 9, G>([
        TensorRank1([DIAG, OFF, OFF, OFF]),
        TensorRank1([OFF, DIAG, OFF, OFF]),
        TensorRank1([OFF, OFF, DIAG, OFF]),
        TensorRank1([OFF, OFF, OFF, DIAG]),
    ]);
const STANDARD_GRADIENT_OPERATORS: StandardGradientOperators<M, O, P> =
    TensorRank1List2D::<M, 9, O, P>([
        TensorRank1List([
            TensorRank1([-2.0, -2.0, -2.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([2.0, 0.0, 0.0]),
            tensor_rank_1_zero(),
            TensorRank1([0.0, 2.0, 0.0]),
            TensorRank1([0.0, 0.0, 2.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            TensorRank1([2.0, 0.0, 0.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([-2.0, -2.0, -2.0]),
            TensorRank1([0.0, 2.0, 0.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([0.0, 0.0, 2.0]),
            tensor_rank_1_zero(),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([0.0, 2.0, 0.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([2.0, 0.0, 0.0]),
            TensorRank1([-2.0, -2.0, -2.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([0.0, 0.0, 2.0]),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([0.0, 0.0, 2.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([-2.0, -2.0, -2.0]),
            TensorRank1([2.0, 0.0, 0.0]),
            TensorRank1([0.0, 2.0, 0.0]),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([-TWO_THIRDS, -2.0, -2.0]),
            TensorRank1([FOUR_THIRDS, 2.0, 0.0]),
            TensorRank1([-TWO_THIRDS, 0.0, 0.0]),
            TensorRank1([-TWO_THIRDS, 0.0, 0.0]),
            TensorRank1([FOUR_THIRDS, 0.0, 2.0]),
            TensorRank1([-TWO_THIRDS, 0.0, 0.0]),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([-TWO_THIRDS, -TWO_THIRDS, -TWO_THIRDS]),
            TensorRank1([FOUR_THIRDS, FOUR_THIRDS, -TWO_THIRDS]),
            TensorRank1([-TWO_THIRDS, -TWO_THIRDS, -TWO_THIRDS]),
            TensorRank1([-TWO_THIRDS, -TWO_THIRDS, -TWO_THIRDS]),
            TensorRank1([FOUR_THIRDS, -TWO_THIRDS, FOUR_THIRDS]),
            TensorRank1([-TWO_THIRDS, FOUR_THIRDS, FOUR_THIRDS]),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([0.0, 0.0, -TWO_THIRDS]),
            TensorRank1([0.0, 0.0, -TWO_THIRDS]),
            TensorRank1([0.0, 0.0, -TWO_THIRDS]),
            TensorRank1([-2.0, -2.0, -TWO_THIRDS]),
            TensorRank1([2.0, 0.0, FOUR_THIRDS]),
            TensorRank1([0.0, 2.0, FOUR_THIRDS]),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([0.0, -FOUR_THIRDS, -2.0]),
            TensorRank1([0.0, TWO_THIRDS, 0.0]),
            TensorRank1([0.0, TWO_THIRDS, 0.0]),
            TensorRank1([-2.0, -FOUR_THIRDS, 0.0]),
            TensorRank1([2.0, TWO_THIRDS, 2.0]),
            TensorRank1([0.0, TWO_THIRDS, 0.0]),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([0.0, -2.0, -FOUR_THIRDS]),
            TensorRank1([2.0, 2.0, TWO_THIRDS]),
            TensorRank1([-2.0, 0.0, -FOUR_THIRDS]),
            TensorRank1([0.0, 0.0, TWO_THIRDS]),
            TensorRank1([0.0, 0.0, TWO_THIRDS]),
            TensorRank1([0.0, 0.0, TWO_THIRDS]),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([0.0, -TWO_THIRDS, 0.0]),
            TensorRank1([2.0, FOUR_THIRDS, 0.0]),
            TensorRank1([-2.0, -TWO_THIRDS, -2.0]),
            TensorRank1([0.0, -TWO_THIRDS, 0.0]),
            TensorRank1([0.0, -TWO_THIRDS, 0.0]),
            TensorRank1([0.0, FOUR_THIRDS, 2.0]),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([TWO_THIRDS, 0.0, 0.0]),
            TensorRank1([TWO_THIRDS, 0.0, 0.0]),
            TensorRank1([-FOUR_THIRDS, 0.0, -2.0]),
            TensorRank1([-FOUR_THIRDS, -2.0, 0.0]),
            TensorRank1([TWO_THIRDS, 0.0, 0.0]),
            TensorRank1([TWO_THIRDS, 2.0, 2.0]),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([TWO_THIRDS, -FOUR_THIRDS, -FOUR_THIRDS]),
            TensorRank1([TWO_THIRDS, TWO_THIRDS, TWO_THIRDS]),
            TensorRank1([-FOUR_THIRDS, TWO_THIRDS, -FOUR_THIRDS]),
            TensorRank1([-FOUR_THIRDS, -FOUR_THIRDS, TWO_THIRDS]),
            TensorRank1([TWO_THIRDS, TWO_THIRDS, TWO_THIRDS]),
            TensorRank1([TWO_THIRDS, TWO_THIRDS, TWO_THIRDS]),
        ]),
    ]);
const STANDARD_GRADIENT_OPERATORS_TRANSPOSED: StandardGradientOperatorsTransposed<M, O, P> =
    TensorRank1List2D::<M, 9, P, O>([
        TensorRank1List([
            TensorRank1([-2.0, -2.0, -2.0]),
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
        TensorRank1List([
            tensor_rank_1_zero(),
            TensorRank1([2.0, 0.0, 0.0]),
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
        TensorRank1List([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([0.0, 2.0, 0.0]),
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
        TensorRank1List([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([0.0, 0.0, 2.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
        ]),
        TensorRank1List([
            TensorRank1([2.0, 0.0, 0.0]),
            TensorRank1([-2.0, -2.0, -2.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([-TWO_THIRDS, -2.0, -2.0]),
            TensorRank1([-TWO_THIRDS, -TWO_THIRDS, -TWO_THIRDS]),
            TensorRank1([0.0, 0.0, -TWO_THIRDS]),
            TensorRank1([0.0, -FOUR_THIRDS, -2.0]),
            TensorRank1([0.0, -2.0, -FOUR_THIRDS]),
            TensorRank1([0.0, -TWO_THIRDS, 0.0]),
            TensorRank1([TWO_THIRDS, 0.0, 0.0]),
            TensorRank1([TWO_THIRDS, -FOUR_THIRDS, -FOUR_THIRDS]),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            TensorRank1([0.0, 2.0, 0.0]),
            TensorRank1([2.0, 0.0, 0.0]),
            tensor_rank_1_zero(),
            TensorRank1([FOUR_THIRDS, 2.0, 0.0]),
            TensorRank1([FOUR_THIRDS, FOUR_THIRDS, -TWO_THIRDS]),
            TensorRank1([0.0, 0.0, -TWO_THIRDS]),
            TensorRank1([0.0, TWO_THIRDS, 0.0]),
            TensorRank1([2.0, 2.0, TWO_THIRDS]),
            TensorRank1([2.0, FOUR_THIRDS, 0.0]),
            TensorRank1([TWO_THIRDS, 0.0, 0.0]),
            TensorRank1([TWO_THIRDS, TWO_THIRDS, TWO_THIRDS]),
        ]),
        TensorRank1List([
            TensorRank1([0.0, 2.0, 0.0]),
            tensor_rank_1_zero(),
            TensorRank1([-2.0, -2.0, -2.0]),
            tensor_rank_1_zero(),
            TensorRank1([-TWO_THIRDS, 0.0, 0.0]),
            TensorRank1([-TWO_THIRDS, -TWO_THIRDS, -TWO_THIRDS]),
            TensorRank1([0.0, 0.0, -TWO_THIRDS]),
            TensorRank1([0.0, TWO_THIRDS, 0.0]),
            TensorRank1([-2.0, 0.0, -FOUR_THIRDS]),
            TensorRank1([-2.0, -TWO_THIRDS, -2.0]),
            TensorRank1([-FOUR_THIRDS, 0.0, -2.0]),
            TensorRank1([-FOUR_THIRDS, TWO_THIRDS, -FOUR_THIRDS]),
        ]),
        TensorRank1List([
            TensorRank1([0.0, 0.0, 2.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([-2.0, -2.0, -2.0]),
            TensorRank1([-TWO_THIRDS, 0.0, 0.0]),
            TensorRank1([-TWO_THIRDS, -TWO_THIRDS, -TWO_THIRDS]),
            TensorRank1([-2.0, -2.0, -TWO_THIRDS]),
            TensorRank1([-2.0, -FOUR_THIRDS, 0.0]),
            TensorRank1([0.0, 0.0, TWO_THIRDS]),
            TensorRank1([0.0, -TWO_THIRDS, 0.0]),
            TensorRank1([-FOUR_THIRDS, -2.0, 0.0]),
            TensorRank1([-FOUR_THIRDS, -FOUR_THIRDS, TWO_THIRDS]),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            TensorRank1([0.0, 0.0, 2.0]),
            tensor_rank_1_zero(),
            TensorRank1([2.0, 0.0, 0.0]),
            TensorRank1([FOUR_THIRDS, 0.0, 2.0]),
            TensorRank1([FOUR_THIRDS, -TWO_THIRDS, FOUR_THIRDS]),
            TensorRank1([2.0, 0.0, FOUR_THIRDS]),
            TensorRank1([2.0, TWO_THIRDS, 2.0]),
            TensorRank1([0.0, 0.0, TWO_THIRDS]),
            TensorRank1([0.0, -TWO_THIRDS, 0.0]),
            TensorRank1([TWO_THIRDS, 0.0, 0.0]),
            TensorRank1([TWO_THIRDS, TWO_THIRDS, TWO_THIRDS]),
        ]),
        TensorRank1List([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            TensorRank1([0.0, 0.0, 2.0]),
            TensorRank1([0.0, 2.0, 0.0]),
            TensorRank1([-TWO_THIRDS, 0.0, 0.0]),
            TensorRank1([-TWO_THIRDS, FOUR_THIRDS, FOUR_THIRDS]),
            TensorRank1([0.0, 2.0, FOUR_THIRDS]),
            TensorRank1([0.0, TWO_THIRDS, 0.0]),
            TensorRank1([0.0, 0.0, TWO_THIRDS]),
            TensorRank1([0.0, FOUR_THIRDS, 2.0]),
            TensorRank1([TWO_THIRDS, 2.0, 2.0]),
            TensorRank1([TWO_THIRDS, TWO_THIRDS, TWO_THIRDS]),
        ]),
    ]);

pub struct Tetrahedron<C> {
    constitutive_models: [C; G],
    integration_weights: Scalars<G>,
    projected_gradient_vectors: ProjectedGradientVectors<G, N>,
}

impl<'a, C> FiniteElement<'a, C, G, N> for Tetrahedron<C>
where
    C: Constitutive<'a>,
{
    fn new(
        constitutive_model_parameters: Parameters<'a>,
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
    ) -> Self {
        Self {
            constitutive_models: std::array::from_fn(|_| <C>::new(constitutive_model_parameters)),
            integration_weights: Self::calculate_reference_jacobians(&reference_nodal_coordinates)
                * INTEGRATION_WEIGHT,
            projected_gradient_vectors: Self::calculate_projected_gradient_vectors(
                &reference_nodal_coordinates,
            ),
        }
    }
}

impl<'a, C> CompositeElement<'a, C, G, M, N, O, P, Q> for Tetrahedron<C>
where
    C: Constitutive<'a>,
{
    fn calculate_inverse_normalized_projection_matrix() -> NormalizedProjectionMatrix<Q> {
        INVERSE_NORMALIED_PROJECTION_MATRIX
    }
    fn calculate_projected_gradient_vectors(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
    ) -> ProjectedGradientVectors<G, N> {
        let parametric_gradient_operators = STANDARD_GRADIENT_OPERATORS
            .iter()
            .map(|standard_gradient_operator| {
                reference_nodal_coordinates * standard_gradient_operator
            })
            .collect::<ParametricGradientOperators<P>>();
        let reference_jacobians_subelements =
            Self::calculate_reference_jacobians_subelements(reference_nodal_coordinates);
        let inverse_projection_matrix =
            Self::calculate_inverse_projection_matrix(&reference_jacobians_subelements);
        SHAPE_FUNCTIONS_AT_INTEGRATION_POINTS
            .iter()
            .map(|shape_functions_at_integration_point| {
                STANDARD_GRADIENT_OPERATORS_TRANSPOSED
                    .iter()
                    .map(|standard_gradient_operators_a| {
                        SHAPE_FUNCTION_INTEGRALS
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
    fn calculate_reference_jacobians_subelements(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<N>,
    ) -> Scalars<P> {
        STANDARD_GRADIENT_OPERATORS
            .iter()
            .map(|standard_gradient_operator| {
                reference_nodal_coordinates * standard_gradient_operator
            })
            .collect::<ParametricGradientOperators<P>>()
            .iter()
            .map(|parametric_gradient_operator| parametric_gradient_operator.determinant())
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

impl<'a, C> ElasticFiniteElement<'a, C, G, N> for Tetrahedron<C>
where
    C: Elastic<'a>,
{
    fn calculate_deformations(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> DeformationGradients<G> {
        self.calculate_deformation_gradients(nodal_coordinates)
    }
    fn calculate_nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalForces<N>, ConstitutiveError> {
        self.calculate_nodal_forces_composite_element(nodal_coordinates)
    }
    fn calculate_nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalStiffnesses<N>, ConstitutiveError> {
        self.calculate_nodal_stiffnesses_composite_element(nodal_coordinates)
    }
}

impl<'a, C> ElasticCompositeElement<'a, C, G, M, N, O, P, Q> for Tetrahedron<C> where C: Elastic<'a> {}

impl<'a, C> HyperelasticFiniteElement<'a, C, G, N> for Tetrahedron<C>
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

impl<'a, C> HyperelasticCompositeElement<'a, C, G, M, N, O, P, Q> for Tetrahedron<C> where
    C: Hyperelastic<'a>
{
}

impl<'a, C> ViscoelasticFiniteElement<'a, C, G, N> for Tetrahedron<C>
where
    C: Viscoelastic<'a>,
{
    fn calculate_nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalForces<N>, ConstitutiveError> {
        self.calculate_nodal_forces_composite_element(nodal_coordinates, nodal_velocities)
    }
    fn calculate_nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalStiffnesses<N>, ConstitutiveError> {
        self.calculate_nodal_stiffnesses_composite_element(nodal_coordinates, nodal_velocities)
    }
}

impl<'a, C> ViscoelasticCompositeElement<'a, C, G, M, N, O, P, Q> for Tetrahedron<C> where
    C: Viscoelastic<'a>
{
}

impl<'a, C> ElasticHyperviscousFiniteElement<'a, C, G, N> for Tetrahedron<C>
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

impl<'a, C> ElasticHyperviscousCompositeElement<'a, C, G, M, N, O, P, Q> for Tetrahedron<C> where
    C: ElasticHyperviscous<'a>
{
}

impl<'a, C> HyperviscoelasticFiniteElement<'a, C, G, N> for Tetrahedron<C>
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

impl<'a, C> HyperviscoelasticCompositeElement<'a, C, G, M, N, O, P, Q> for Tetrahedron<C> where
    C: Hyperviscoelastic<'a>
{
}
