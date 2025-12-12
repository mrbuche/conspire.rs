#[cfg(test)]
mod test;

use crate::{
    fem::{
        GradientVectors, ReferenceNodalCoordinates, ShapeFunctionsGradients,
        StandardGradientOperators,
        block::element::{Element, FiniteElement},
    },
    math::{Scalar, Scalars, Tensor},
    mechanics::Coordinate,
};

#[cfg(test)]
use crate::fem::{ShapeFunctions, ShapeFunctionsAtIntegrationPoints};

const G: usize = 8;
const M: usize = 3;
const N: usize = 8;
const P: usize = N;

#[cfg(test)]
const Q: usize = N;

const SQRT_3: Scalar = 1.732_050_807_568_877_2;

pub type Hexahedron = Element<G, N>;

impl From<ReferenceNodalCoordinates<N>> for Hexahedron {
    fn from(reference_nodal_coordinates: ReferenceNodalCoordinates<N>) -> Self {
        let (gradient_vectors, integration_weights) = Self::initialize(reference_nodal_coordinates);
        Self {
            gradient_vectors,
            integration_weights,
        }
    }
}

impl FiniteElement<G, N> for Hexahedron {
    fn reference() -> ReferenceNodalCoordinates<N> {
        ReferenceNodalCoordinates::const_from([
            Coordinate::const_from([-1.0, -1.0, -1.0]),
            Coordinate::const_from([1.0, -1.0, -1.0]),
            Coordinate::const_from([1.0, 1.0, -1.0]),
            Coordinate::const_from([-1.0, 1.0, -1.0]),
            Coordinate::const_from([-1.0, -1.0, 1.0]),
            Coordinate::const_from([1.0, -1.0, 1.0]),
            Coordinate::const_from([1.0, 1.0, 1.0]),
            Coordinate::const_from([-1.0, 1.0, 1.0]),
        ])
    }
    fn reset(&mut self) {
        let (gradient_vectors, integration_weights) = Self::initialize(Self::reference());
        self.gradient_vectors = gradient_vectors;
        self.integration_weights = integration_weights;
    }
}

impl Hexahedron {
    fn initialize(
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
    ) -> (GradientVectors<G, N>, Scalars<G>) {
        let standard_gradient_operators = Self::standard_gradient_operators();
        let gradient_vectors = standard_gradient_operators
            .iter()
            .map(|standard_gradient_operator| {
                (&reference_nodal_coordinates * standard_gradient_operator).inverse_transpose()
                    * standard_gradient_operator
            })
            .collect();
        let integration_weights = standard_gradient_operators
            .iter()
            .map(|standard_gradient_operator| {
                (&reference_nodal_coordinates * standard_gradient_operator).determinant()
                    * Self::integration_weight()
            })
            .collect();
        (gradient_vectors, integration_weights)
    }
    const fn integration_point(point: usize) -> [Scalar; M] {
        match point {
            0 => [-SQRT_3 / 3.0, -SQRT_3 / 3.0, -SQRT_3 / 3.0],
            1 => [-SQRT_3 / 3.0, -SQRT_3 / 3.0, SQRT_3 / 3.0],
            2 => [-SQRT_3 / 3.0, SQRT_3 / 3.0, -SQRT_3 / 3.0],
            3 => [-SQRT_3 / 3.0, SQRT_3 / 3.0, SQRT_3 / 3.0],
            4 => [SQRT_3 / 3.0, -SQRT_3 / 3.0, -SQRT_3 / 3.0],
            5 => [SQRT_3 / 3.0, -SQRT_3 / 3.0, SQRT_3 / 3.0],
            6 => [SQRT_3 / 3.0, SQRT_3 / 3.0, -SQRT_3 / 3.0],
            7 => [SQRT_3 / 3.0, SQRT_3 / 3.0, SQRT_3 / 3.0],
            _ => panic!(),
        }
    }
    const fn integration_weight() -> Scalar {
        1.0
    }
    #[cfg(test)]
    const fn shape_functions([xi_1, xi_2, xi_3]: [Scalar; M]) -> ShapeFunctions<N> {
        ShapeFunctions::const_from([
            (1.0 - xi_1) * (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 - xi_1) * (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 - xi_1) * (1.0 - xi_2) * (1.0 + xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 - xi_2) * (1.0 + xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 + xi_2) * (1.0 + xi_3) / 8.0,
            (1.0 - xi_1) * (1.0 + xi_2) * (1.0 + xi_3) / 8.0,
        ])
    }
    #[cfg(test)]
    const fn shape_functions_at_integration_points() -> ShapeFunctionsAtIntegrationPoints<G, Q> {
        ShapeFunctionsAtIntegrationPoints::const_from([
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
    const fn shape_functions_gradients(
        [xi_1, xi_2, xi_3]: [Scalar; M],
    ) -> ShapeFunctionsGradients<M, N> {
        ShapeFunctionsGradients::const_from([
            Coordinate::const_from([
                -(1.0 - xi_2) * (1.0 - xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 - xi_2) / 8.0,
            ]),
            Coordinate::const_from([
                (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 - xi_2) / 8.0,
            ]),
            Coordinate::const_from([
                (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 + xi_2) / 8.0,
            ]),
            Coordinate::const_from([
                -(1.0 + xi_2) * (1.0 - xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 + xi_2) / 8.0,
            ]),
            Coordinate::const_from([
                -(1.0 - xi_2) * (1.0 + xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 - xi_2) / 8.0,
            ]),
            Coordinate::const_from([
                (1.0 - xi_2) * (1.0 + xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 - xi_2) / 8.0,
            ]),
            Coordinate::const_from([
                (1.0 + xi_2) * (1.0 + xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 + xi_2) / 8.0,
            ]),
            Coordinate::const_from([
                -(1.0 + xi_2) * (1.0 + xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 + xi_2) / 8.0,
            ]),
        ])
    }
    const fn standard_gradient_operators() -> StandardGradientOperators<M, N, P> {
        StandardGradientOperators::const_from([
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
