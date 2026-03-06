#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        FRAC_1_SQRT_3, FiniteElement, ParametricCoordinate, ParametricCoordinates,
        ParametricReference, ShapeFunctions, ShapeFunctionsGradients,
        linear::{LinearElement, LinearFiniteElement, M},
    },
    math::{Scalar, ScalarList},
};

const G: usize = 8;
const N: usize = 5;
const P: usize = N;

pub type Pyramid = LinearElement<G, N>;

impl FiniteElement<G, M, N, P> for Pyramid {
    fn integration_points() -> ParametricCoordinates<G, M> {
        integration_points_and_weights().0
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
        integration_points_and_weights().1
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        let bottom = bottom(xi_3);
        [
            ((1.0 - xi_1) * (1.0 - xi_2) - xi_3 + xi_1 * xi_2 * xi_3 / bottom) / 4.0,
            ((1.0 + xi_1) * (1.0 - xi_2) - xi_3 - xi_1 * xi_2 * xi_3 / bottom) / 4.0,
            ((1.0 + xi_1) * (1.0 + xi_2) - xi_3 + xi_1 * xi_2 * xi_3 / bottom) / 4.0,
            ((1.0 - xi_1) * (1.0 + xi_2) - xi_3 - xi_1 * xi_2 * xi_3 / bottom) / 4.0,
            xi_3,
        ]
        .into()
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        let bottom = bottom(xi_3);
        let bottom_squared = bottom * bottom;
        [
            [
                (-(1.0 - xi_2) + xi_2 * xi_3 / bottom) / 4.0,
                (-(1.0 - xi_1) + xi_1 * xi_3 / bottom) / 4.0,
                (-1.0 + xi_1 * xi_2 / bottom_squared) / 4.0,
            ],
            [
                ((1.0 - xi_2) - xi_2 * xi_3 / bottom) / 4.0,
                (-(1.0 + xi_1) - xi_1 * xi_3 / bottom) / 4.0,
                (-1.0 - xi_1 * xi_2 / bottom_squared) / 4.0,
            ],
            [
                ((1.0 + xi_2) + xi_2 * xi_3 / bottom) / 4.0,
                ((1.0 + xi_1) + xi_1 * xi_3 / bottom) / 4.0,
                (-1.0 + xi_1 * xi_2 / bottom_squared) / 4.0,
            ],
            [
                (-(1.0 + xi_2) - xi_2 * xi_3 / bottom) / 4.0,
                ((1.0 - xi_1) - xi_1 * xi_3 / bottom) / 4.0,
                (-1.0 - xi_1 * xi_2 / bottom_squared) / 4.0,
            ],
            [0.0, 0.0, 1.0],
        ]
        .into()
    }
}

fn bottom(xi_3: Scalar) -> Scalar {
    const SMALL: Scalar = 4e1 * f64::EPSILON;
    if (1.0 - xi_3).abs() > SMALL {
        1.0 - xi_3
    } else {
        SMALL
    }
}

fn integration_points_and_weights() -> (ParametricCoordinates<G, M>, ScalarList<G>) {
    const X: [Scalar; 2] = [0.455_848_155_988_775, 0.877_485_177_344_559];
    const B: [Scalar; 2] = [0.100_785_882_079_825, 0.232_547_451_253_508];
    const U1_2D: [Scalar; 4] = [-FRAC_1_SQRT_3, FRAC_1_SQRT_3, FRAC_1_SQRT_3, -FRAC_1_SQRT_3];
    const U2_2D: [Scalar; 4] = [-FRAC_1_SQRT_3, -FRAC_1_SQRT_3, FRAC_1_SQRT_3, FRAC_1_SQRT_3];
    const W_2D: [Scalar; 4] = [1.0; _];
    let mut points = [[0.0; M]; G];
    let mut weights = [0.0; G];
    let mut i = 0;
    X.into_iter().zip(B).for_each(|(x, b)| {
        U1_2D
            .into_iter()
            .zip(U2_2D)
            .zip(W_2D)
            .for_each(|((u1, u2), w)| {
                points[i][0] = x * u1;
                points[i][1] = x * u2;
                points[i][2] = 1.0 - x;
                weights[i] = w * b;
                i += 1;
            })
    });
    (points.into(), weights.into())
}

impl LinearFiniteElement<G, N> for Pyramid {}
