#[cfg(test)]
mod test;

use super::*;
use crate::math::{
    tensor_rank_1, tensor_rank_1_list, tensor_rank_1_list_2d, tensor_rank_1_zero, tensor_rank_2,
    tensor_rank_2_list,
};

const G: usize = 4;
const M: usize = 3;
const N: usize = 10;
const O: usize = 10;
const P: usize = 12;
const Q: usize = 4;

const INTEGRATION_WEIGHT: Scalar = 1.0 / 24.0;

const INVERSE_NORMALIED_PROJECTION_MATRIX: NormalizedProjectionMatrix<Q> = tensor_rank_2([
    tensor_rank_1([4.0 / 640.0, -1.0 / 640.0, -1.0 / 640.0, -1.0 / 640.0]),
    tensor_rank_1([-1.0 / 640.0, 4.0 / 640.0, -1.0 / 640.0, -1.0 / 640.0]),
    tensor_rank_1([-1.0 / 640.0, -1.0 / 640.0, 4.0 / 640.0, -1.0 / 640.0]),
    tensor_rank_1([-1.0 / 640.0, -1.0 / 640.0, -1.0 / 640.0, 4.0 / 640.0]),
]);
const SHAPE_FUNCTION_INTEGRALS: ShapeFunctionIntegrals<P, Q> = tensor_rank_1_list([
    tensor_rank_1([200.0, 40.0, 40.0, 40.0]),
    tensor_rank_1([40.0, 200.0, 40.0, 40.0]),
    tensor_rank_1([40.0, 40.0, 200.0, 40.0]),
    tensor_rank_1([40.0, 40.0, 40.0, 200.0]),
    tensor_rank_1([30.0, 70.0, 30.0, 30.0]),
    tensor_rank_1([10.0, 50.0, 50.0, 50.0]),
    tensor_rank_1([30.0, 30.0, 30.0, 70.0]),
    tensor_rank_1([50.0, 50.0, 10.0, 50.0]),
    tensor_rank_1([50.0, 50.0, 50.0, 10.0]),
    tensor_rank_1([30.0, 30.0, 70.0, 30.0]),
    tensor_rank_1([50.0, 10.0, 50.0, 50.0]),
    tensor_rank_1([70.0, 30.0, 30.0, 30.0]),
]);
const SHAPE_FUNCTION_INTEGRALS_PRODUCTS: ShapeFunctionIntegralsProducts<P, Q> =
    tensor_rank_2_list([
        tensor_rank_2([
            tensor_rank_1([128.0, 24.0, 24.0, 24.0]),
            tensor_rank_1([24.0, 8.0, 4.0, 4.0]),
            tensor_rank_1([24.0, 4.0, 8.0, 4.0]),
            tensor_rank_1([24.0, 4.0, 4.0, 8.0]),
        ]),
        tensor_rank_2([
            tensor_rank_1([8.0, 24.0, 4.0, 4.0]),
            tensor_rank_1([24.0, 128.0, 24.0, 24.0]),
            tensor_rank_1([4.0, 24.0, 8.0, 4.0]),
            tensor_rank_1([4.0, 24.0, 4.0, 8.0]),
        ]),
        tensor_rank_2([
            tensor_rank_1([8.0, 4.0, 24.0, 4.0]),
            tensor_rank_1([4.0, 8.0, 24.0, 4.0]),
            tensor_rank_1([24.0, 24.0, 128.0, 24.0]),
            tensor_rank_1([4.0, 4.0, 24.0, 8.0]),
        ]),
        tensor_rank_2([
            tensor_rank_1([8.0, 4.0, 4.0, 24.0]),
            tensor_rank_1([4.0, 8.0, 4.0, 24.0]),
            tensor_rank_1([4.0, 4.0, 8.0, 24.0]),
            tensor_rank_1([24.0, 24.0, 24.0, 128.0]),
        ]),
        tensor_rank_2([
            tensor_rank_1([7.0, 13.0, 5.0, 5.0]),
            tensor_rank_1([13.0, 31.0, 13.0, 13.0]),
            tensor_rank_1([5.0, 13.0, 7.0, 5.0]),
            tensor_rank_1([5.0, 13.0, 5.0, 7.0]),
        ]),
        tensor_rank_2([
            tensor_rank_1([1.0, 3.0, 3.0, 3.0]),
            tensor_rank_1([3.0, 17.0, 15.0, 15.0]),
            tensor_rank_1([3.0, 15.0, 17.0, 15.0]),
            tensor_rank_1([3.0, 15.0, 15.0, 17.0]),
        ]),
        tensor_rank_2([
            tensor_rank_1([7.0, 5.0, 5.0, 13.0]),
            tensor_rank_1([5.0, 7.0, 5.0, 13.0]),
            tensor_rank_1([5.0, 5.0, 7.0, 13.0]),
            tensor_rank_1([13.0, 13.0, 13.0, 31.0]),
        ]),
        tensor_rank_2([
            tensor_rank_1([17.0, 15.0, 3.0, 15.0]),
            tensor_rank_1([15.0, 17.0, 3.0, 15.0]),
            tensor_rank_1([3.0, 3.0, 1.0, 3.0]),
            tensor_rank_1([15.0, 15.0, 3.0, 17.0]),
        ]),
        tensor_rank_2([
            tensor_rank_1([17.0, 15.0, 15.0, 3.0]),
            tensor_rank_1([15.0, 17.0, 15.0, 3.0]),
            tensor_rank_1([15.0, 15.0, 17.0, 3.0]),
            tensor_rank_1([3.0, 3.0, 3.0, 1.0]),
        ]),
        tensor_rank_2([
            tensor_rank_1([7.0, 5.0, 13.0, 5.0]),
            tensor_rank_1([5.0, 7.0, 13.0, 5.0]),
            tensor_rank_1([13.0, 13.0, 31.0, 13.0]),
            tensor_rank_1([5.0, 5.0, 13.0, 7.0]),
        ]),
        tensor_rank_2([
            tensor_rank_1([17.0, 3.0, 15.0, 15.0]),
            tensor_rank_1([3.0, 1.0, 3.0, 3.0]),
            tensor_rank_1([15.0, 3.0, 17.0, 15.0]),
            tensor_rank_1([15.0, 3.0, 15.0, 17.0]),
        ]),
        tensor_rank_2([
            tensor_rank_1([31.0, 13.0, 13.0, 13.0]),
            tensor_rank_1([13.0, 7.0, 5.0, 5.0]),
            tensor_rank_1([13.0, 5.0, 7.0, 5.0]),
            tensor_rank_1([13.0, 5.0, 5.0, 7.0]),
        ]),
    ]);
const DIAG: Scalar = 0.585_410_196_624_968_5;
const OFF: Scalar = 0.138_196_601_125_010_5;
const SHAPE_FUNCTIONS_AT_INTEGRATION_POINTS: ShapeFunctionsAtIntegrationPoints<G, Q> =
    tensor_rank_1_list([
        tensor_rank_1([DIAG, OFF, OFF, OFF]),
        tensor_rank_1([OFF, DIAG, OFF, OFF]),
        tensor_rank_1([OFF, OFF, DIAG, OFF]),
        tensor_rank_1([OFF, OFF, OFF, DIAG]),
    ]);
const STANDARD_GRADIENT_OPERATORS: StandardGradientOperators<M, O, P> = tensor_rank_1_list_2d([
    tensor_rank_1_list([
        tensor_rank_1([-2.0, -2.0, -2.0]),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1([2.0, 0.0, 0.0]),
        tensor_rank_1_zero(),
        tensor_rank_1([0.0, 2.0, 0.0]),
        tensor_rank_1([0.0, 0.0, 2.0]),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
    ]),
    tensor_rank_1_list([
        tensor_rank_1_zero(),
        tensor_rank_1([2.0, 0.0, 0.0]),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1([-2.0, -2.0, -2.0]),
        tensor_rank_1([0.0, 2.0, 0.0]),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1([0.0, 0.0, 2.0]),
        tensor_rank_1_zero(),
    ]),
    tensor_rank_1_list([
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1([0.0, 2.0, 0.0]),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1([2.0, 0.0, 0.0]),
        tensor_rank_1([-2.0, -2.0, -2.0]),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1([0.0, 0.0, 2.0]),
    ]),
    tensor_rank_1_list([
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1([0.0, 0.0, 2.0]),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1([-2.0, -2.0, -2.0]),
        tensor_rank_1([2.0, 0.0, 0.0]),
        tensor_rank_1([0.0, 2.0, 0.0]),
    ]),
    tensor_rank_1_list([
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1([-2.0 / 3.0, -2.0, -2.0]),
        tensor_rank_1([4.0 / 3.0, 2.0, 0.0]),
        tensor_rank_1([-2.0 / 3.0, 0.0, 0.0]),
        tensor_rank_1([-2.0 / 3.0, 0.0, 0.0]),
        tensor_rank_1([4.0 / 3.0, 0.0, 2.0]),
        tensor_rank_1([-2.0 / 3.0, 0.0, 0.0]),
    ]),
    tensor_rank_1_list([
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1([-2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0]),
        tensor_rank_1([4.0 / 3.0, 4.0 / 3.0, -2.0 / 3.0]),
        tensor_rank_1([-2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0]),
        tensor_rank_1([-2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0]),
        tensor_rank_1([4.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0]),
        tensor_rank_1([-2.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0]),
    ]),
    tensor_rank_1_list([
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1([0.0, 0.0, -2.0 / 3.0]),
        tensor_rank_1([0.0, 0.0, -2.0 / 3.0]),
        tensor_rank_1([0.0, 0.0, -2.0 / 3.0]),
        tensor_rank_1([-2.0, -2.0, -2.0 / 3.0]),
        tensor_rank_1([2.0, 0.0, 4.0 / 3.0]),
        tensor_rank_1([0.0, 2.0, 4.0 / 3.0]),
    ]),
    tensor_rank_1_list([
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1([0.0, -4.0 / 3.0, -2.0]),
        tensor_rank_1([0.0, 2.0 / 3.0, 0.0]),
        tensor_rank_1([0.0, 2.0 / 3.0, 0.0]),
        tensor_rank_1([-2.0, -4.0 / 3.0, 0.0]),
        tensor_rank_1([2.0, 2.0 / 3.0, 2.0]),
        tensor_rank_1([0.0, 2.0 / 3.0, 0.0]),
    ]),
    tensor_rank_1_list([
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1([0.0, -2.0, -4.0 / 3.0]),
        tensor_rank_1([2.0, 2.0, 2.0 / 3.0]),
        tensor_rank_1([-2.0, 0.0, -4.0 / 3.0]),
        tensor_rank_1([0.0, 0.0, 2.0 / 3.0]),
        tensor_rank_1([0.0, 0.0, 2.0 / 3.0]),
        tensor_rank_1([0.0, 0.0, 2.0 / 3.0]),
    ]),
    tensor_rank_1_list([
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1([0.0, -2.0 / 3.0, 0.0]),
        tensor_rank_1([2.0, 4.0 / 3.0, 0.0]),
        tensor_rank_1([-2.0, -2.0 / 3.0, -2.0]),
        tensor_rank_1([0.0, -2.0 / 3.0, 0.0]),
        tensor_rank_1([0.0, -2.0 / 3.0, 0.0]),
        tensor_rank_1([0.0, 4.0 / 3.0, 2.0]),
    ]),
    tensor_rank_1_list([
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1([2.0 / 3.0, 0.0, 0.0]),
        tensor_rank_1([2.0 / 3.0, 0.0, 0.0]),
        tensor_rank_1([-4.0 / 3.0, 0.0, -2.0]),
        tensor_rank_1([-4.0 / 3.0, -2.0, 0.0]),
        tensor_rank_1([2.0 / 3.0, 0.0, 0.0]),
        tensor_rank_1([2.0 / 3.0, 2.0, 2.0]),
    ]),
    tensor_rank_1_list([
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1_zero(),
        tensor_rank_1([2.0 / 3.0, -4.0 / 3.0, -4.0 / 3.0]),
        tensor_rank_1([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]),
        tensor_rank_1([-4.0 / 3.0, 2.0 / 3.0, -4.0 / 3.0]),
        tensor_rank_1([-4.0 / 3.0, -4.0 / 3.0, 2.0 / 3.0]),
        tensor_rank_1([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]),
        tensor_rank_1([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]),
    ]),
]);
const STANDARD_GRADIENT_OPERATORS_TRANSPOSED: StandardGradientOperatorsTransposed<M, O, P> =
    tensor_rank_1_list_2d([
        tensor_rank_1_list([
            tensor_rank_1([-2.0, -2.0, -2.0]),
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
        tensor_rank_1_list([
            tensor_rank_1_zero(),
            tensor_rank_1([2.0, 0.0, 0.0]),
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
        tensor_rank_1_list([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1([0.0, 2.0, 0.0]),
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
        tensor_rank_1_list([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1([0.0, 0.0, 2.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
        ]),
        tensor_rank_1_list([
            tensor_rank_1([2.0, 0.0, 0.0]),
            tensor_rank_1([-2.0, -2.0, -2.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1([-2.0 / 3.0, -2.0, -2.0]),
            tensor_rank_1([-2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0]),
            tensor_rank_1([0.0, 0.0, -2.0 / 3.0]),
            tensor_rank_1([0.0, -4.0 / 3.0, -2.0]),
            tensor_rank_1([0.0, -2.0, -4.0 / 3.0]),
            tensor_rank_1([0.0, -2.0 / 3.0, 0.0]),
            tensor_rank_1([2.0 / 3.0, 0.0, 0.0]),
            tensor_rank_1([2.0 / 3.0, -4.0 / 3.0, -4.0 / 3.0]),
        ]),
        tensor_rank_1_list([
            tensor_rank_1_zero(),
            tensor_rank_1([0.0, 2.0, 0.0]),
            tensor_rank_1([2.0, 0.0, 0.0]),
            tensor_rank_1_zero(),
            tensor_rank_1([4.0 / 3.0, 2.0, 0.0]),
            tensor_rank_1([4.0 / 3.0, 4.0 / 3.0, -2.0 / 3.0]),
            tensor_rank_1([0.0, 0.0, -2.0 / 3.0]),
            tensor_rank_1([0.0, 2.0 / 3.0, 0.0]),
            tensor_rank_1([2.0, 2.0, 2.0 / 3.0]),
            tensor_rank_1([2.0, 4.0 / 3.0, 0.0]),
            tensor_rank_1([2.0 / 3.0, 0.0, 0.0]),
            tensor_rank_1([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]),
        ]),
        tensor_rank_1_list([
            tensor_rank_1([0.0, 2.0, 0.0]),
            tensor_rank_1_zero(),
            tensor_rank_1([-2.0, -2.0, -2.0]),
            tensor_rank_1_zero(),
            tensor_rank_1([-2.0 / 3.0, 0.0, 0.0]),
            tensor_rank_1([-2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0]),
            tensor_rank_1([0.0, 0.0, -2.0 / 3.0]),
            tensor_rank_1([0.0, 2.0 / 3.0, 0.0]),
            tensor_rank_1([-2.0, 0.0, -4.0 / 3.0]),
            tensor_rank_1([-2.0, -2.0 / 3.0, -2.0]),
            tensor_rank_1([-4.0 / 3.0, 0.0, -2.0]),
            tensor_rank_1([-4.0 / 3.0, 2.0 / 3.0, -4.0 / 3.0]),
        ]),
        tensor_rank_1_list([
            tensor_rank_1([0.0, 0.0, 2.0]),
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1([-2.0, -2.0, -2.0]),
            tensor_rank_1([-2.0 / 3.0, 0.0, 0.0]),
            tensor_rank_1([-2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0]),
            tensor_rank_1([-2.0, -2.0, -2.0 / 3.0]),
            tensor_rank_1([-2.0, -4.0 / 3.0, 0.0]),
            tensor_rank_1([0.0, 0.0, 2.0 / 3.0]),
            tensor_rank_1([0.0, -2.0 / 3.0, 0.0]),
            tensor_rank_1([-4.0 / 3.0, -2.0, 0.0]),
            tensor_rank_1([-4.0 / 3.0, -4.0 / 3.0, 2.0 / 3.0]),
        ]),
        tensor_rank_1_list([
            tensor_rank_1_zero(),
            tensor_rank_1([0.0, 0.0, 2.0]),
            tensor_rank_1_zero(),
            tensor_rank_1([2.0, 0.0, 0.0]),
            tensor_rank_1([4.0 / 3.0, 0.0, 2.0]),
            tensor_rank_1([4.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0]),
            tensor_rank_1([2.0, 0.0, 4.0 / 3.0]),
            tensor_rank_1([2.0, 2.0 / 3.0, 2.0]),
            tensor_rank_1([0.0, 0.0, 2.0 / 3.0]),
            tensor_rank_1([0.0, -2.0 / 3.0, 0.0]),
            tensor_rank_1([2.0 / 3.0, 0.0, 0.0]),
            tensor_rank_1([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]),
        ]),
        tensor_rank_1_list([
            tensor_rank_1_zero(),
            tensor_rank_1_zero(),
            tensor_rank_1([0.0, 0.0, 2.0]),
            tensor_rank_1([0.0, 2.0, 0.0]),
            tensor_rank_1([-2.0 / 3.0, 0.0, 0.0]),
            tensor_rank_1([-2.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0]),
            tensor_rank_1([0.0, 2.0, 4.0 / 3.0]),
            tensor_rank_1([0.0, 2.0 / 3.0, 0.0]),
            tensor_rank_1([0.0, 0.0, 2.0 / 3.0]),
            tensor_rank_1([0.0, 4.0 / 3.0, 2.0]),
            tensor_rank_1([2.0 / 3.0, 2.0, 2.0]),
            tensor_rank_1([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]),
        ]),
    ]);

pub type Tetrahedron<C> = Element<C, G, M, N, O>;

impl<'a, C> FiniteElement<'a, C, G, N> for Tetrahedron<C>
where
    C: Constitutive<'a>,
{
    fn new(
        constitutive_model_parameters: Parameters<'a>,
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
    ) -> Self {
        // let (operator, jacobian) = (reference_nodal_coordinates * STANDARD_GRADIENT_OPERATOR)
        //     .inverse_transpose_and_determinant();
        // Self {
        //     constitutive_models: from_fn(|_| <C>::new(constitutive_model_parameters)),
        //     gradient_vectors: tensor_rank_1_list_2d([operator * STANDARD_GRADIENT_OPERATOR]),
        //     integration_weights: tensor_rank_0_list([jacobian * INTEGRATION_WEIGHT]),
        // }
        todo!()
    }
}

// pub struct Tetrahedron<C> {
//     constitutive_models: [C; G],
//     integration_weights: Scalars<G>,
//     projected_gradient_vectors: ProjectedGradientVectors<G, N>,
// }

// impl<'a, C> FiniteElement<'a, C, G, N> for Tetrahedron<C>
// where
//     C: Constitutive<'a>,
// {
//     fn constitutive_models(&self) -> &[C; G] {
//         &self.constitutive_models
//     }
//     fn integration_weights(&self) -> &Scalars<G> {
//         &self.integration_weights
//     }
//     fn new(
//         constitutive_model_parameters: Parameters<'a>,
//         reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
//     ) -> Self {
//         Self {
//             constitutive_models: std::array::from_fn(|_| <C>::new(constitutive_model_parameters)),
//             integration_weights: Self::reference_jacobians(&reference_nodal_coordinates)
//                 * INTEGRATION_WEIGHT,
//             projected_gradient_vectors: Self::projected_gradient_vectors(
//                 &reference_nodal_coordinates,
//             ),
//         }
//     }
// }

// impl<'a, C> CompositeElement<'a, C, G, M, N, O, P, Q> for Tetrahedron<C>
// where
//     C: Constitutive<'a>,
// {
//     fn inverse_normalized_projection_matrix() -> NormalizedProjectionMatrix<Q> {
//         INVERSE_NORMALIED_PROJECTION_MATRIX
//     }
//     fn projected_gradient_vectors(
//         reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
//     ) -> ProjectedGradientVectors<G, N> {
//         let parametric_gradient_operators = STANDARD_GRADIENT_OPERATORS
//             .iter()
//             .map(|standard_gradient_operator| {
//                 reference_nodal_coordinates * standard_gradient_operator
//             })
//             .collect::<ParametricGradientOperators<P>>();
//         let reference_jacobians_subelements =
//             Self::reference_jacobians_subelements(reference_nodal_coordinates);
//         let inverse_projection_matrix =
//             Self::inverse_projection_matrix(&reference_jacobians_subelements);
//         SHAPE_FUNCTIONS_AT_INTEGRATION_POINTS
//             .iter()
//             .map(|shape_functions_at_integration_point| {
//                 STANDARD_GRADIENT_OPERATORS_TRANSPOSED
//                     .iter()
//                     .map(|standard_gradient_operators_a| {
//                         SHAPE_FUNCTION_INTEGRALS
//                             .iter()
//                             .zip(
//                                 standard_gradient_operators_a.iter().zip(
//                                     parametric_gradient_operators
//                                         .iter()
//                                         .zip(reference_jacobians_subelements.iter()),
//                                 ),
//                             )
//                             .map(
//                                 |(
//                                     shape_function_integral,
//                                     (
//                                         standard_gradient_operator,
//                                         (
//                                             parametric_gradient_operator,
//                                             reference_jacobian_subelement,
//                                         ),
//                                     ),
//                                 )| {
//                                     (parametric_gradient_operator.inverse_transpose()
//                                         * standard_gradient_operator)
//                                         * reference_jacobian_subelement
//                                         * (shape_functions_at_integration_point
//                                             * (&inverse_projection_matrix
//                                                 * shape_function_integral))
//                                 },
//                             )
//                             .sum()
//                     })
//                     .collect()
//             })
//             .collect()
//     }
//     fn reference_jacobians_subelements(
//         reference_nodal_coordinates: &ReferenceNodalCoordinates<N>,
//     ) -> Scalars<P> {
//         STANDARD_GRADIENT_OPERATORS
//             .iter()
//             .map(|standard_gradient_operator| {
//                 reference_nodal_coordinates * standard_gradient_operator
//             })
//             .collect::<ParametricGradientOperators<P>>()
//             .iter()
//             .map(|parametric_gradient_operator| parametric_gradient_operator.determinant())
//             .collect()
//     }
//     fn shape_function_integrals() -> ShapeFunctionIntegrals<P, Q> {
//         SHAPE_FUNCTION_INTEGRALS
//     }
//     fn shape_function_integrals_products() -> ShapeFunctionIntegralsProducts<P, Q> {
//         SHAPE_FUNCTION_INTEGRALS_PRODUCTS
//     }
//     fn shape_functions_at_integration_points() -> ShapeFunctionsAtIntegrationPoints<G, Q> {
//         SHAPE_FUNCTIONS_AT_INTEGRATION_POINTS
//     }
//     fn standard_gradient_operators() -> StandardGradientOperators<M, O, P> {
//         STANDARD_GRADIENT_OPERATORS
//     }
//     fn standard_gradient_operators_transposed() -> StandardGradientOperatorsTransposed<M, O, P> {
//         STANDARD_GRADIENT_OPERATORS_TRANSPOSED
//     }
//     fn get_constitutive_models(&self) -> &[C; G] {
//         &self.constitutive_models
//     }
//     fn get_integration_weights(&self) -> &Scalars<G> {
//         &self.integration_weights
//     }
//     fn get_projected_gradient_vectors(&self) -> &ProjectedGradientVectors<G, N> {
//         &self.projected_gradient_vectors
//     }
// }

// impl<'a, C> ElasticFiniteElement<'a, C, G, N> for Tetrahedron<C>
// where
//     C: Elastic<'a>,
// {
//     fn nodal_forces(
//         &self,
//         nodal_coordinates: &NodalCoordinates<N>,
//     ) -> Result<NodalForces<N>, ConstitutiveError> {
//         self.nodal_forces_composite_element(nodal_coordinates)
//     }
//     fn nodal_stiffnesses(
//         &self,
//         nodal_coordinates: &NodalCoordinates<N>,
//     ) -> Result<NodalStiffnesses<N>, ConstitutiveError> {
//         self.nodal_stiffnesses_composite_element(nodal_coordinates)
//     }
// }

// impl<'a, C> ElasticCompositeElement<'a, C, G, M, N, O, P, Q> for Tetrahedron<C> where C: Elastic<'a> {}

// impl<'a, C> HyperelasticFiniteElement<'a, C, G, N> for Tetrahedron<C>
// where
//     C: Hyperelastic<'a>,
// {
//     fn helmholtz_free_energy(
//         &self,
//         nodal_coordinates: &NodalCoordinates<N>,
//     ) -> Result<Scalar, ConstitutiveError> {
//         self.helmholtz_free_energy_composite_element(nodal_coordinates)
//     }
// }

// impl<'a, C> HyperelasticCompositeElement<'a, C, G, M, N, O, P, Q> for Tetrahedron<C> where
//     C: Hyperelastic<'a>
// {
// }

// impl<'a, C> ViscoelasticFiniteElement<'a, C, G, N> for Tetrahedron<C>
// where
//     C: Viscoelastic<'a>,
// {
//     fn nodal_forces(
//         &self,
//         nodal_coordinates: &NodalCoordinates<N>,
//         nodal_velocities: &NodalVelocities<N>,
//     ) -> Result<NodalForces<N>, ConstitutiveError> {
//         self.nodal_forces_composite_element(nodal_coordinates, nodal_velocities)
//     }
//     fn nodal_stiffnesses(
//         &self,
//         nodal_coordinates: &NodalCoordinates<N>,
//         nodal_velocities: &NodalVelocities<N>,
//     ) -> Result<NodalStiffnesses<N>, ConstitutiveError> {
//         self.nodal_stiffnesses_composite_element(nodal_coordinates, nodal_velocities)
//     }
// }

// impl<'a, C> ViscoelasticCompositeElement<'a, C, G, M, N, O, P, Q> for Tetrahedron<C> where
//     C: Viscoelastic<'a>
// {
// }

// impl<'a, C> ElasticHyperviscousFiniteElement<'a, C, G, N> for Tetrahedron<C>
// where
//     C: ElasticHyperviscous<'a>,
// {
//     fn viscous_dissipation(
//         &self,
//         nodal_coordinates: &NodalCoordinates<N>,
//         nodal_velocities: &NodalVelocities<N>,
//     ) -> Result<Scalar, ConstitutiveError> {
//         self.viscous_dissipation_composite_element(nodal_coordinates, nodal_velocities)
//     }
//     fn dissipation_potential(
//         &self,
//         nodal_coordinates: &NodalCoordinates<N>,
//         nodal_velocities: &NodalVelocities<N>,
//     ) -> Result<Scalar, ConstitutiveError> {
//         self.dissipation_potential_composite_element(nodal_coordinates, nodal_velocities)
//     }
// }

// impl<'a, C> ElasticHyperviscousCompositeElement<'a, C, G, M, N, O, P, Q> for Tetrahedron<C> where
//     C: ElasticHyperviscous<'a>
// {
// }

// impl<'a, C> HyperviscoelasticFiniteElement<'a, C, G, N> for Tetrahedron<C>
// where
//     C: Hyperviscoelastic<'a>,
// {
//     fn helmholtz_free_energy(
//         &self,
//         nodal_coordinates: &NodalCoordinates<N>,
//     ) -> Result<Scalar, ConstitutiveError> {
//         self.helmholtz_free_energy_composite_element(nodal_coordinates)
//     }
// }

// impl<'a, C> HyperviscoelasticCompositeElement<'a, C, G, M, N, O, P, Q> for Tetrahedron<C> where
//     C: Hyperviscoelastic<'a>
// {
// }
