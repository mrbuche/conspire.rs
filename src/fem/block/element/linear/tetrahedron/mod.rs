#[cfg(test)]
mod test;

use super::*;
use crate::{
    math::{tensor_rank_0_list, tensor_rank_1, tensor_rank_1_list, tensor_rank_1_list_2d},
    mechanics::Scalar,
};

const G: usize = 1;
const M: usize = 3;
const N: usize = 4;
const P: usize = G;

#[cfg(test)]
const Q: usize = N;

pub type Tetrahedron<'a, C> = Element<'a, C, G, N>;

impl<'a, C> FiniteElement<'a, C, G, N> for Tetrahedron<'a, C> {
    fn new(
        constitutive_model: &'a C,
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
    ) -> Self {
        let (gradient_vectors, integration_weights) = Self::initialize(reference_nodal_coordinates);
        Self {
            constitutive_model,
            gradient_vectors,
            integration_weights,
        }
    }
    fn reference() -> ReferenceNodalCoordinates<N> {
        tensor_rank_1_list([
            tensor_rank_1([0.0, 0.0, 0.0]),
            tensor_rank_1([1.0, 0.0, 0.0]),
            tensor_rank_1([0.0, 1.0, 0.0]),
            tensor_rank_1([0.0, 0.0, 1.0]),
        ])
    }
    fn reset(&mut self) {
        let (gradient_vectors, integration_weights) = Self::initialize(Self::reference());
        self.gradient_vectors = gradient_vectors;
        self.integration_weights = integration_weights;
    }
}

impl<'a, C> Tetrahedron<'a, C> {
    fn initialize(
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
    ) -> (GradientVectors<G, N>, Scalars<G>) {
        let standard_gradient_operator = &Self::standard_gradient_operators()[0];
        let (operator, jacobian) = (reference_nodal_coordinates * standard_gradient_operator)
            .inverse_transpose_and_determinant();
        let gradient_vectors = tensor_rank_1_list_2d([operator * standard_gradient_operator]);
        let integration_weights = tensor_rank_0_list([jacobian * Self::integration_weight()]);
        (gradient_vectors, integration_weights)
    }
    const fn integration_weight() -> Scalar {
        1.0 / 6.0
    }
    #[cfg(test)]
    const fn shape_functions_at_integration_points() -> ShapeFunctionsAtIntegrationPoints<G, Q> {
        tensor_rank_1_list([tensor_rank_1([0.25; Q])])
    }
    const fn standard_gradient_operators() -> StandardGradientOperators<M, N, P> {
        tensor_rank_1_list_2d([tensor_rank_1_list([
            tensor_rank_1([-1.0, -1.0, -1.0]),
            tensor_rank_1([1.0, 0.0, 0.0]),
            tensor_rank_1([0.0, 1.0, 0.0]),
            tensor_rank_1([0.0, 0.0, 1.0]),
        ])])
    }
}
