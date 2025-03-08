#[cfg(test)]
mod test;

use super::*;
use crate::{
    constitutive::{Constitutive, Parameters},
    math::{tensor_rank_0_list, tensor_rank_1, tensor_rank_1_list, tensor_rank_1_list_2d},
    mechanics::Scalar,
};
use std::array::from_fn;

const G: usize = 1;
const M: usize = 3;
const N: usize = 4;
const O: usize = 4;
const P: usize = 1;

pub type Tetrahedron<C> = Element<C, G, M, N, O>;

impl<'a, C> FiniteElement<'a, C, G, N> for Tetrahedron<C>
where
    C: Constitutive<'a>,
{
    fn new(
        constitutive_model_parameters: Parameters<'a>,
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
    ) -> Self {
        let standard_gradient_operator = &Self::standard_gradient_operators()[0];
        let (operator, jacobian) = (reference_nodal_coordinates * standard_gradient_operator)
        .inverse_transpose_and_determinant();
        Self {
            constitutive_models: from_fn(|_| <C>::new(constitutive_model_parameters)),
            gradient_vectors: tensor_rank_1_list_2d(
                [operator * standard_gradient_operator],
            ),
            integration_weights: tensor_rank_0_list([jacobian * Self::integration_weight()]),
        }
    }
}

impl<'a, C> Tetrahedron<C>
where
    C: Constitutive<'a>,
{
    const fn integration_weight() -> Scalar {
        1.0 / 6.0
    }
    const fn standard_gradient_operators() -> StandardGradientOperators<M, O, P> {
        tensor_rank_1_list_2d([
            tensor_rank_1_list([
                tensor_rank_1([-1.0, -1.0, -1.0]),
                tensor_rank_1([1.0, 0.0, 0.0]),
                tensor_rank_1([0.0, 1.0, 0.0]),
                tensor_rank_1([0.0, 0.0, 1.0]),
        ])
        ])
    }
}
