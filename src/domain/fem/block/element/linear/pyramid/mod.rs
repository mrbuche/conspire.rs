#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        FiniteElement, ParametricCoordinate, ParametricCoordinates, ParametricReference,
        ShapeFunctions, ShapeFunctionsGradients,
        linear::{LinearElement, LinearFiniteElement, M},
    },
    math::{Scalar, ScalarList},
};

const G: usize = 5;
const N: usize = 5;

pub type Pyramid = LinearElement<G, N>;

impl FiniteElement<G, M, N> for Pyramid {
    fn integration_points() -> ParametricCoordinates<G, M> {
        [
            [-0.5, 0.0, 1.0 / 6.0],
            [0.5, 0.0, 1.0 / 6.0],
            [0.0, -0.5, 1.0 / 6.0],
            [0.0, 0.5, 1.0 / 6.0],
            [0.0, 0.0, 0.25],
        ]
        .into()
    }
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        .into()
    }
    fn parametric_weights() -> ScalarList<G> {
        [5.0 / 27.0, 5.0 / 27.0, 5.0 / 27.0, 5.0 / 27.0, 16.0 / 27.0].into()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        let bottom = bottom(xi_3);
        [
            (1.0 - xi_1 - xi_3) * (1.0 - xi_2 - xi_3) / bottom / 4.0,
            (1.0 + xi_1 - xi_3) * (1.0 - xi_2 - xi_3) / bottom / 4.0,
            (1.0 + xi_1 - xi_3) * (1.0 + xi_2 - xi_3) / bottom / 4.0,
            (1.0 - xi_1 - xi_3) * (1.0 + xi_2 - xi_3) / bottom / 4.0,
            xi_3,
        ]
        .into()
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        let bottom = bottom(xi_3);
        [
            [
                -(1.0 - xi_2 - xi_3) / bottom / 4.0,
                -(1.0 - xi_1 - xi_3) / bottom / 4.0,
                (-(1.0 - xi_2 - xi_3) - (1.0 - xi_1 - xi_3)
                    + (1.0 - xi_1 - xi_3) * (1.0 - xi_2 - xi_3) / bottom)
                    / bottom
                    / 4.0,
            ],
            [
                (1.0 - xi_2 - xi_3) / bottom / 4.0,
                -(1.0 + xi_1 - xi_3) / bottom / 4.0,
                (-(1.0 - xi_2 - xi_3) - (1.0 + xi_1 - xi_3)
                    + (1.0 + xi_1 - xi_3) * (1.0 - xi_2 - xi_3) / bottom)
                    / bottom
                    / 4.0,
            ],
            [
                (1.0 + xi_2 - xi_3) / bottom / 4.0,
                (1.0 + xi_1 - xi_3) / bottom / 4.0,
                (-(1.0 + xi_2 - xi_3) - (1.0 + xi_1 - xi_3)
                    + (1.0 + xi_1 - xi_3) * (1.0 + xi_2 - xi_3) / bottom)
                    / bottom
                    / 4.0,
            ],
            [
                -(1.0 + xi_2 - xi_3) / bottom / 4.0,
                (1.0 - xi_1 - xi_3) / bottom / 4.0,
                (-(1.0 + xi_2 - xi_3) - (1.0 - xi_1 - xi_3)
                    + (1.0 - xi_1 - xi_3) * (1.0 + xi_2 - xi_3) / bottom)
                    / bottom
                    / 4.0,
            ],
            [0.0, 0.0, 1.0],
        ]
        .into()
    }
}

fn bottom(xi_3: Scalar) -> Scalar {
    const SMALL: Scalar = 1e1 * f64::EPSILON;
    if (1.0 - xi_3).abs() > SMALL {
        1.0 - xi_3
    } else {
        SMALL
    }
}

impl LinearFiniteElement<G, N> for Pyramid {}
