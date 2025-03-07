#[cfg(test)]
mod test;

use super::*;
use crate::math::{tensor_rank_0_list, tensor_rank_1, tensor_rank_1_list, tensor_rank_1_list_2d, tensor_rank_2_list, TensorArray};

const G: usize = 1;
const M: usize = 3;
const N: usize = 4;
const O: usize = 4;

const INTEGRATION_WEIGHT: Scalar = ONE_SIXTH;

const STANDARD_GRADIENT_OPERATOR: StandardGradientOperator<M, O> = tensor_rank_1_list([
    tensor_rank_1([-1.0, -1.0, -1.0]),
    tensor_rank_1([1.0, 0.0, 0.0]),
    tensor_rank_1([0.0, 1.0, 0.0]),
    tensor_rank_1([0.0, 0.0, 1.0]),
]);

pub type Tetrahedron<C> = FooFiniteElement<C, G, M, N, O>;

impl<'a, C> FiniteElementMethods<'a, C, N> for Tetrahedron<C>
where
    C: Constitutive<'a>
{
    fn new(
        constitutive_model_parameters: Parameters<'a>,
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
    ) -> Self {
        let (operator, jacobian) = (reference_nodal_coordinates * STANDARD_GRADIENT_OPERATOR).inverse_transpose_and_determinant();
        Self {
            constitutive_models: std::array::from_fn(|_| <C>::new(constitutive_model_parameters)),
            gradient_vectors: tensor_rank_1_list_2d([operator * STANDARD_GRADIENT_OPERATOR]),
            integration_weights: tensor_rank_0_list([jacobian * INTEGRATION_WEIGHT]),
        }
    }
}
