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
const M: usize = 2;
const N: usize = 3;
const P: usize = 1;

#[cfg(test)]
const Q: usize = 3;

pub type Triangle<C> = SurfaceElement<C, G, N>;

impl<'a, C> SurfaceFiniteElement<'a, C, G, N> for Triangle<C>
where
    C: Constitutive<'a>,
{
    fn new(
        constitutive_model_parameters: Parameters<'a>,
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
        thickness: &Scalar,
    ) -> Self {
        // let standard_gradient_operator = &Self::standard_gradient_operators()[0];
        // let (operator, jacobian) = (reference_nodal_coordinates * standard_gradient_operator)
        //     .inverse_transpose_and_determinant();
        // Self {
        //     constitutive_models: from_fn(|_| <C>::new(constitutive_model_parameters)),
        //     gradient_vectors: tensor_rank_1_list_2d([operator * standard_gradient_operator]),
        //     integration_weights: tensor_rank_0_list([jacobian * Self::integration_weight()]),
        // }
        todo!()
    }
}

impl<'a, C> Triangle<C>
where
    C: Constitutive<'a>,
{
    const fn integration_weight() -> Scalar {
        1.0 / 2.0
    }
    #[cfg(test)]
    const fn shape_functions_at_integration_points() -> ShapeFunctionsAtIntegrationPoints<G, Q> {
        tensor_rank_1_list([tensor_rank_1([1.0 / 3.0; Q])])
    }
    const fn standard_gradient_operators() -> StandardGradientOperators<M, N, P> {
        tensor_rank_1_list_2d([tensor_rank_1_list([
            tensor_rank_1([-1.0, -1.0]),
            tensor_rank_1([1.0, 0.0]),
            tensor_rank_1([0.0, 1.0]),
        ])])
    }
}