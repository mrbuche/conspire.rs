#[cfg(test)]
mod test;

use crate::{
    fem::{
        GradientVectors, ReferenceNodalCoordinates, StandardGradientOperators,
        block::element::{Element, FiniteElement},
    },
    math::{Scalar, Scalars},
};

#[cfg(test)]
use crate::fem::ShapeFunctionsAtIntegrationPoints;

const G: usize = 1;
const M: usize = 3;
const N: usize = 4;
const P: usize = G;

#[cfg(test)]
const Q: usize = N;

pub type Tetrahedron = Element<G, N>;

impl FiniteElement<G, N> for Tetrahedron {
    fn initialize(
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
    ) -> (GradientVectors<G, N>, Scalars<G>) {
        let standard_gradient_operator = &Self::standard_gradient_operators()[0];
        let (operator, jacobian) = (reference_nodal_coordinates * standard_gradient_operator)
            .inverse_transpose_and_determinant();
        let gradient_vectors = GradientVectors::from([operator * standard_gradient_operator]);
        let integration_weights =
            Scalars::<G>::const_from_array([jacobian * Self::integration_weight()]);
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
        1.0 / 6.0
    }
    const fn reference() -> ReferenceNodalCoordinates<N> {
        ReferenceNodalCoordinates::<N>::const_from_array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    }
    #[cfg(test)]
    const fn shape_functions_at_integration_points() -> ShapeFunctionsAtIntegrationPoints<G, Q> {
        ShapeFunctionsAtIntegrationPoints::<G, Q>::const_from_array([[0.25; Q]])
    }
    const fn standard_gradient_operators() -> StandardGradientOperators<M, N, P> {
        StandardGradientOperators::<M, N, P>::const_from_array([[
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]])
    }
}
