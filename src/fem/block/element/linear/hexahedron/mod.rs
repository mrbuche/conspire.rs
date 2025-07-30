#[cfg(test)]
mod test;

use super::*;
use crate::{
    constitutive::{Constitutive, Parameters},
    math::{tensor_rank_1, tensor_rank_1_list, tensor_rank_1_list_2d},
    mechanics::Scalar,
};
use std::array::from_fn;

const G: usize = 8;
const M: usize = 3;
const N: usize = 8;
const P: usize = N;

#[cfg(test)]
const Q: usize = N;

const SQRT_3: Scalar = 1.732_050_807_568_877_2;

pub type Hexahedron<C> = Element<C, G, N>;

impl<C, Y> FiniteElement<C, G, N, Y> for Hexahedron<C>
where
    C: Constitutive<Y>,
    Y: Parameters,
{
    fn new(
        constitutive_model_parameters: Y,
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
    ) -> Self {
        let standard_gradient_operators = Self::standard_gradient_operators();
        let gradient_vectors = standard_gradient_operators
            .iter()
            .map(|standard_gradient_operator| {
                (&reference_nodal_coordinates * standard_gradient_operator).inverse_transpose()
                    * standard_gradient_operator
            })
            .collect();
        let integration_weights = standard_gradient_operators
            .iter()
            .map(|standard_gradient_operator| {
                (&reference_nodal_coordinates * standard_gradient_operator).determinant()
                    * Self::integration_weight()
            })
            .collect();
        Self {
            constitutive_models: from_fn(|_| <C>::new(constitutive_model_parameters)),
            gradient_vectors,
            integration_weights,
        }
    }
}

impl<C> Hexahedron<C> {
    const fn integration_point(point: usize) -> [Scalar; M] {
        match point {
            0 => [-SQRT_3 / 3.0, -SQRT_3 / 3.0, -SQRT_3 / 3.0],
            1 => [-SQRT_3 / 3.0, -SQRT_3 / 3.0, SQRT_3 / 3.0],
            2 => [-SQRT_3 / 3.0, SQRT_3 / 3.0, -SQRT_3 / 3.0],
            3 => [-SQRT_3 / 3.0, SQRT_3 / 3.0, SQRT_3 / 3.0],
            4 => [SQRT_3 / 3.0, -SQRT_3 / 3.0, -SQRT_3 / 3.0],
            5 => [SQRT_3 / 3.0, -SQRT_3 / 3.0, SQRT_3 / 3.0],
            6 => [SQRT_3 / 3.0, SQRT_3 / 3.0, -SQRT_3 / 3.0],
            7 => [SQRT_3 / 3.0, SQRT_3 / 3.0, SQRT_3 / 3.0],
            _ => panic!(),
        }
    }
    const fn integration_weight() -> Scalar {
        1.0
    }
    #[cfg(test)]
    const fn shape_functions([xi_1, xi_2, xi_3]: [Scalar; M]) -> ShapeFunctions<N> {
        tensor_rank_1([
            (1.0 - xi_1) * (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 - xi_1) * (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 - xi_1) * (1.0 - xi_2) * (1.0 + xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 - xi_2) * (1.0 + xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 + xi_2) * (1.0 + xi_3) / 8.0,
            (1.0 - xi_1) * (1.0 + xi_2) * (1.0 + xi_3) / 8.0,
        ])
    }
    #[cfg(test)]
    const fn shape_functions_at_integration_points() -> ShapeFunctionsAtIntegrationPoints<G, Q> {
        tensor_rank_1_list([
            Self::shape_functions(Self::integration_point(0)),
            Self::shape_functions(Self::integration_point(1)),
            Self::shape_functions(Self::integration_point(2)),
            Self::shape_functions(Self::integration_point(3)),
            Self::shape_functions(Self::integration_point(4)),
            Self::shape_functions(Self::integration_point(5)),
            Self::shape_functions(Self::integration_point(6)),
            Self::shape_functions(Self::integration_point(7)),
        ])
    }
    const fn shape_functions_gradients(
        [xi_1, xi_2, xi_3]: [Scalar; M],
    ) -> ShapeFunctionsGradients<M, N> {
        tensor_rank_1_list([
            tensor_rank_1([
                -(1.0 - xi_2) * (1.0 - xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 - xi_2) / 8.0,
            ]),
            tensor_rank_1([
                (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 - xi_2) / 8.0,
            ]),
            tensor_rank_1([
                (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 + xi_2) / 8.0,
            ]),
            tensor_rank_1([
                -(1.0 + xi_2) * (1.0 - xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 + xi_2) / 8.0,
            ]),
            tensor_rank_1([
                -(1.0 - xi_2) * (1.0 + xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 - xi_2) / 8.0,
            ]),
            tensor_rank_1([
                (1.0 - xi_2) * (1.0 + xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 - xi_2) / 8.0,
            ]),
            tensor_rank_1([
                (1.0 + xi_2) * (1.0 + xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 + xi_2) / 8.0,
            ]),
            tensor_rank_1([
                -(1.0 + xi_2) * (1.0 + xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 + xi_2) / 8.0,
            ]),
        ])
    }
    const fn standard_gradient_operators() -> StandardGradientOperators<M, N, P> {
        tensor_rank_1_list_2d([
            Self::shape_functions_gradients(Self::integration_point(0)),
            Self::shape_functions_gradients(Self::integration_point(1)),
            Self::shape_functions_gradients(Self::integration_point(2)),
            Self::shape_functions_gradients(Self::integration_point(3)),
            Self::shape_functions_gradients(Self::integration_point(4)),
            Self::shape_functions_gradients(Self::integration_point(5)),
            Self::shape_functions_gradients(Self::integration_point(6)),
            Self::shape_functions_gradients(Self::integration_point(7)),
        ])
    }
}
