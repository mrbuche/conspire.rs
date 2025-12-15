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

const G: usize = 5;
const M: usize = 3;
const N: usize = 5;
const P: usize = N;

#[cfg(test)]
const Q: usize = N;

pub type Pyramid = Element<G, N>;

linear_finite_element!(
    Pyramid,
    5,
    [5.0 / 27.0, 5.0 / 27.0, 5.0 / 27.0, 5.0 / 27.0, 16.0 / 27.0]
);

impl Pyramid {
    const fn integration_point(point: usize) -> [Scalar; M] {
        match point {
            0 => [-0.5, 0.0, 1.0 / 6.0],
            1 => [0.5, 0.0, 1.0 / 6.0],
            2 => [0.0, -0.5, 1.0 / 6.0],
            3 => [0.0, 0.5, 1.0 / 6.0],
            4 => [0.0, 0.0, 0.25],
            _ => panic!(),
        }
    }
    // const fn integration_weight() -> Scalars<G> {
    //     Scalars::<G>::const_from([5.0 / 27.0, 5.0 / 27.0, 5.0 / 27.0, 5.0 / 27.0, 16.0 / 27.0])
    // }
    const fn reference() -> ElementNodalReferenceCoordinates<N> {
        ElementNodalReferenceCoordinates::<N>::const_from([
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    }
    #[cfg(test)]
    const fn shape_functions([xi_1, xi_2, xi_3]: [Scalar; M]) -> [Scalar; N] {
        [
            (1.0 - xi_1) * (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 - xi_1) * (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 + xi_3) / 2.0,
        ]
    }
    #[cfg(test)]
    const fn shape_functions_at_integration_points() -> ShapeFunctionsAtIntegrationPoints<G, Q> {
        ShapeFunctionsAtIntegrationPoints::<G, Q>::const_from([
            Self::shape_functions(Self::integration_point(0)),
            Self::shape_functions(Self::integration_point(1)),
            Self::shape_functions(Self::integration_point(2)),
            Self::shape_functions(Self::integration_point(3)),
            Self::shape_functions(Self::integration_point(4)),
        ])
    }
    const fn shape_functions_gradients([xi_1, xi_2, xi_3]: [Scalar; M]) -> [[Scalar; M]; N] {
        [
            [
                -(1.0 - xi_2) * (1.0 - xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 - xi_2) / 8.0,
            ],
            [
                (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 - xi_2) / 8.0,
            ],
            [
                (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 + xi_2) / 8.0,
            ],
            [
                -(1.0 + xi_2) * (1.0 - xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 + xi_2) / 8.0,
            ],
            [
                0.0,
                0.0,
                0.5,
            ],
        ]
    }
    // const fn standard_gradient_operators() -> StandardGradientOperators<M, N, P> {
    //     StandardGradientOperators::<M, N, P>::const_from([
    //         Self::shape_functions_gradients(Self::integration_point(0)),
    //         Self::shape_functions_gradients(Self::integration_point(1)),
    //         Self::shape_functions_gradients(Self::integration_point(2)),
    //         Self::shape_functions_gradients(Self::integration_point(3)),
    //         Self::shape_functions_gradients(Self::integration_point(4)),
    //     ])
    // }
}
