#[cfg(test)]
mod test;

use crate::{
    fem::{
        StandardGradientOperators,
        block::element::{
            Element, ElementNodalReferenceCoordinates, FiniteElement, GradientVectors,
            linear::linear_finite_element,
        },
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

linear_finite_element!(Tetrahedron);

impl Tetrahedron {
    const fn integration_weight() -> Scalar {
        1.0 / 6.0
    }
    const fn reference() -> ElementNodalReferenceCoordinates<N> {
        ElementNodalReferenceCoordinates::<N>::const_from([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    }
    #[cfg(test)]
    const fn shape_functions_at_integration_points() -> ShapeFunctionsAtIntegrationPoints<G, Q> {
        ShapeFunctionsAtIntegrationPoints::<G, Q>::const_from([[0.25; Q]])
    }
    const fn standard_gradient_operators() -> StandardGradientOperators<M, N, P> {
        StandardGradientOperators::<M, N, P>::const_from([[
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]])
    }
}
