#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        Element, ElementNodalReferenceCoordinates, FiniteElement, GradientVectors,
        StandardGradientOperators, linear::linear_finite_element,
    },
    math::{Scalar, Scalars},
};

#[cfg(test)]
use crate::fem::block::element::ShapeFunctionsAtIntegrationPoints;

const G: usize = 1;
const M: usize = 3;
const N: usize = 4;
const P: usize = G;

#[cfg(test)]
const Q: usize = N;

pub type Tetrahedron = Element<G, N>;

linear_finite_element!(
    Tetrahedron,
    1,
    [1.0 / 6.0]
);

impl Tetrahedron {
    const fn integration_point(point: usize) -> [Scalar; M] {
        match point {
            0 => [0.25; M],
            _ => panic!(),
        }
    }
    // const fn integration_weight() -> Scalars<G> {
    //     Scalars::<G>::const_from([1.0 / 6.0])
    // }
    const fn reference() -> ElementNodalReferenceCoordinates<N> {
        ElementNodalReferenceCoordinates::<N>::const_from([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    }
    #[cfg(test)]
    const fn shape_functions([xi_1, xi_2, xi_3]: [Scalar; M]) -> [Scalar; N] {
        [
            1.0 - xi_1 - xi_2 - xi_3,
            xi_1,
            xi_2,
            xi_3,
        ]
    }
    #[cfg(test)]
    const fn shape_functions_at_integration_points() -> ShapeFunctionsAtIntegrationPoints<G, Q> {
        ShapeFunctionsAtIntegrationPoints::<G, Q>::const_from([
            Self::shape_functions(Self::integration_point(0)),
        ])
    }
    const fn shape_functions_gradients([_xi_1, _xi_2, _xi_3]: [Scalar; M]) -> [[Scalar; M]; N] {
        [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    }
    // const fn standard_gradient_operators() -> StandardGradientOperators<M, N, P> {
    //     StandardGradientOperators::<M, N, P>::const_from([
    //         Self::shape_functions_gradients(Self::integration_point(0)),
    //     ])
    // }
}
