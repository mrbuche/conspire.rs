#[cfg(test)]
mod test;

use crate::{
    fem::{
        GradientVectors, StandardGradientOperators,
        block::element::{
            Element, ElementReferenceNodalCoordinates, FiniteElement, linear::linear_finite_element,
        },
    },
    math::{Scalar, Scalars},
};

#[cfg(test)]
use crate::fem::ShapeFunctionsAtIntegrationPoints;

const G: usize = 8;
const M: usize = 3;
const N: usize = 8;
const P: usize = N;

#[cfg(test)]
const Q: usize = N;

const SQRT_3_OVER_3: Scalar = 0.577_350_269_189_625_8;

pub type Hexahedron = Element<G, N>;

linear_finite_element!(Hexahedron);

impl Hexahedron {
    const fn integration_point(point: usize) -> [Scalar; M] {
        match point {
            0 => [-SQRT_3_OVER_3, -SQRT_3_OVER_3, -SQRT_3_OVER_3],
            1 => [-SQRT_3_OVER_3, -SQRT_3_OVER_3, SQRT_3_OVER_3],
            2 => [-SQRT_3_OVER_3, SQRT_3_OVER_3, -SQRT_3_OVER_3],
            3 => [-SQRT_3_OVER_3, SQRT_3_OVER_3, SQRT_3_OVER_3],
            4 => [SQRT_3_OVER_3, -SQRT_3_OVER_3, -SQRT_3_OVER_3],
            5 => [SQRT_3_OVER_3, -SQRT_3_OVER_3, SQRT_3_OVER_3],
            6 => [SQRT_3_OVER_3, SQRT_3_OVER_3, -SQRT_3_OVER_3],
            7 => [SQRT_3_OVER_3, SQRT_3_OVER_3, SQRT_3_OVER_3],
            _ => panic!(),
        }
    }
    const fn integration_weight() -> Scalar {
        1.0
    }
    const fn reference() -> ElementReferenceNodalCoordinates<N> {
        ElementReferenceNodalCoordinates::<N>::const_from([
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ])
    }
    #[cfg(test)]
    const fn shape_functions([xi_1, xi_2, xi_3]: [Scalar; M]) -> [Scalar; N] {
        [
            (1.0 - xi_1) * (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 - xi_1) * (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 - xi_1) * (1.0 - xi_2) * (1.0 + xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 - xi_2) * (1.0 + xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 + xi_2) * (1.0 + xi_3) / 8.0,
            (1.0 - xi_1) * (1.0 + xi_2) * (1.0 + xi_3) / 8.0,
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
            Self::shape_functions(Self::integration_point(5)),
            Self::shape_functions(Self::integration_point(6)),
            Self::shape_functions(Self::integration_point(7)),
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
                -(1.0 - xi_2) * (1.0 + xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 - xi_2) / 8.0,
            ],
            [
                (1.0 - xi_2) * (1.0 + xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 - xi_2) / 8.0,
            ],
            [
                (1.0 + xi_2) * (1.0 + xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 + xi_2) / 8.0,
            ],
            [
                -(1.0 + xi_2) * (1.0 + xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 + xi_2) / 8.0,
            ],
        ]
    }
    const fn standard_gradient_operators() -> StandardGradientOperators<M, N, P> {
        StandardGradientOperators::<M, N, P>::const_from([
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
