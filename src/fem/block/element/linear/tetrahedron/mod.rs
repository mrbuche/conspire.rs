#[cfg(test)]
mod test;

use crate::{
    fem::{
        GradientVectors, ReferenceNodalCoordinates, StandardGradientOperators,
        block::element::{Element, FiniteElement},
    },
    math::{Scalar, Scalars, TensorRank1List},
};

#[cfg(test)]
use crate::{fem::ShapeFunctionsAtIntegrationPoints, math::TensorRank1};

const G: usize = 1;
const M: usize = 3;
const N: usize = 4;
const P: usize = G;

#[cfg(test)]
const Q: usize = N;

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
        ReferenceNodalCoordinates::foo([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
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
        let standard_gradient_operator = &Self::standard_gradient_operators()[0];
        let (operator, jacobian) = (reference_nodal_coordinates * standard_gradient_operator)
            .inverse_transpose_and_determinant();
        let gradient_vectors = GradientVectors::const_from([operator * standard_gradient_operator]);
        let integration_weights = Scalars::const_from([jacobian * Self::integration_weight()]);
        (gradient_vectors, integration_weights)
    }
    const fn integration_weight() -> Scalar {
        1.0 / 6.0
    }
    #[cfg(test)]
    const fn shape_functions_at_integration_points() -> ShapeFunctionsAtIntegrationPoints<G, Q> {
        ShapeFunctionsAtIntegrationPoints::const_from([TensorRank1::const_from([0.25; Q])])
    }
    const fn standard_gradient_operators() -> StandardGradientOperators<M, N, P> {
        StandardGradientOperators::const_from([TensorRank1List::foo([
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])])
    }
}
