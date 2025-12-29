#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        FRAC_SQRT_3_5, FiniteElement, ParametricCoordinate, ParametricCoordinates,
        ParametricReference, ShapeFunctions, ShapeFunctionsGradients,
        quadratic::{QuadraticElement, QuadraticFiniteElement},
        serendipity::M,
    },
    math::ScalarList,
};

const G: usize = 27;
const N: usize = 20;

pub type Hexahedron = QuadraticElement<G, N>;

impl FiniteElement<G, M, N> for Hexahedron {
    fn integration_points() -> ParametricCoordinates<G, M> {
        let pts = [-FRAC_SQRT_3_5, 0.0, FRAC_SQRT_3_5];
        let mut points = [[0.0; M]; G];
        let mut idx = 0;
        for &z in &pts {
            for &y in &pts {
                for &x in &pts {
                    points[idx] = [x, y, z];
                    idx += 1;
                }
            }
        }
        points.into()
    }
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [0.0, -1.0, -1.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
            [-1.0, 0.0, -1.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [-1.0, 0.0, 1.0],
        ]
        .into()
    }
    fn parametric_weights() -> ScalarList<G> {
        const W1: f64 = 5.0 / 9.0;
        const W2: f64 = 8.0 / 9.0;
        let wts = [W1, W2, W1];
        let mut weights = [0.0; G];
        let mut idx = 0;
        for &wz in &wts {
            for &wy in &wts {
                for &wx in &wts {
                    weights[idx] = wx * wy * wz;
                    idx += 1;
                }
            }
        }
        weights.into()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        [
            0.125 * (1.0 - xi_1) * (1.0 - xi_2) * (1.0 - xi_3) * (-xi_1 - xi_2 - xi_3 - 2.0),
            0.125 * (1.0 + xi_1) * (1.0 - xi_2) * (1.0 - xi_3) * (xi_1 - xi_2 - xi_3 - 2.0),
            0.125 * (1.0 + xi_1) * (1.0 + xi_2) * (1.0 - xi_3) * (xi_1 + xi_2 - xi_3 - 2.0),
            0.125 * (1.0 - xi_1) * (1.0 + xi_2) * (1.0 - xi_3) * (-xi_1 + xi_2 - xi_3 - 2.0),
            0.125 * (1.0 - xi_1) * (1.0 - xi_2) * (1.0 + xi_3) * (-xi_1 - xi_2 + xi_3 - 2.0),
            0.125 * (1.0 + xi_1) * (1.0 - xi_2) * (1.0 + xi_3) * (xi_1 - xi_2 + xi_3 - 2.0),
            0.125 * (1.0 + xi_1) * (1.0 + xi_2) * (1.0 + xi_3) * (xi_1 + xi_2 + xi_3 - 2.0),
            0.125 * (1.0 - xi_1) * (1.0 + xi_2) * (1.0 + xi_3) * (-xi_1 + xi_2 + xi_3 - 2.0),
            0.25 * (1.0 - xi_1 * xi_1) * (1.0 - xi_2) * (1.0 - xi_3),
            0.25 * (1.0 + xi_1) * (1.0 - xi_2 * xi_2) * (1.0 - xi_3),
            0.25 * (1.0 - xi_1 * xi_1) * (1.0 + xi_2) * (1.0 - xi_3),
            0.25 * (1.0 - xi_1) * (1.0 - xi_2 * xi_2) * (1.0 - xi_3),
            0.25 * (1.0 - xi_1) * (1.0 - xi_2) * (1.0 - xi_3 * xi_3),
            0.25 * (1.0 + xi_1) * (1.0 - xi_2) * (1.0 - xi_3 * xi_3),
            0.25 * (1.0 + xi_1) * (1.0 + xi_2) * (1.0 - xi_3 * xi_3),
            0.25 * (1.0 - xi_1) * (1.0 + xi_2) * (1.0 - xi_3 * xi_3),
            0.25 * (1.0 - xi_1 * xi_1) * (1.0 - xi_2) * (1.0 + xi_3),
            0.25 * (1.0 + xi_1) * (1.0 - xi_2 * xi_2) * (1.0 + xi_3),
            0.25 * (1.0 - xi_1 * xi_1) * (1.0 + xi_2) * (1.0 + xi_3),
            0.25 * (1.0 - xi_1) * (1.0 - xi_2 * xi_2) * (1.0 + xi_3),
        ]
        .into()
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        [
            [
                0.125
                    * (-(1.0 - xi_2) * (1.0 - xi_3) * (-xi_1 - xi_2 - xi_3 - 2.0)
                        - (1.0 - xi_1) * (1.0 - xi_2) * (1.0 - xi_3)),
                0.125
                    * (-(1.0 - xi_1) * (1.0 - xi_3) * (-xi_1 - xi_2 - xi_3 - 2.0)
                        - (1.0 - xi_1) * (1.0 - xi_2) * (1.0 - xi_3)),
                0.125
                    * (-(1.0 - xi_1) * (1.0 - xi_2) * (-xi_1 - xi_2 - xi_3 - 2.0)
                        - (1.0 - xi_1) * (1.0 - xi_2) * (1.0 - xi_3)),
            ],
            [
                0.125
                    * ((1.0 - xi_2) * (1.0 - xi_3) * (xi_1 - xi_2 - xi_3 - 2.0)
                        + (1.0 + xi_1) * (1.0 - xi_2) * (1.0 - xi_3)),
                0.125
                    * (-(1.0 + xi_1) * (1.0 - xi_3) * (xi_1 - xi_2 - xi_3 - 2.0)
                        - (1.0 + xi_1) * (1.0 - xi_2) * (1.0 - xi_3)),
                0.125
                    * (-(1.0 + xi_1) * (1.0 - xi_2) * (xi_1 - xi_2 - xi_3 - 2.0)
                        - (1.0 + xi_1) * (1.0 - xi_2) * (1.0 - xi_3)),
            ],
            [
                0.125
                    * ((1.0 + xi_2) * (1.0 - xi_3) * (xi_1 + xi_2 - xi_3 - 2.0)
                        + (1.0 + xi_1) * (1.0 + xi_2) * (1.0 - xi_3)),
                0.125
                    * ((1.0 + xi_1) * (1.0 - xi_3) * (xi_1 + xi_2 - xi_3 - 2.0)
                        + (1.0 + xi_1) * (1.0 + xi_2) * (1.0 - xi_3)),
                0.125
                    * (-(1.0 + xi_1) * (1.0 + xi_2) * (xi_1 + xi_2 - xi_3 - 2.0)
                        - (1.0 + xi_1) * (1.0 + xi_2) * (1.0 - xi_3)),
            ],
            [
                0.125
                    * (-(1.0 + xi_2) * (1.0 - xi_3) * (-xi_1 + xi_2 - xi_3 - 2.0)
                        - (1.0 - xi_1) * (1.0 + xi_2) * (1.0 - xi_3)),
                0.125
                    * ((1.0 - xi_1) * (1.0 - xi_3) * (-xi_1 + xi_2 - xi_3 - 2.0)
                        + (1.0 - xi_1) * (1.0 + xi_2) * (1.0 - xi_3)),
                0.125
                    * (-(1.0 - xi_1) * (1.0 + xi_2) * (-xi_1 + xi_2 - xi_3 - 2.0)
                        - (1.0 - xi_1) * (1.0 + xi_2) * (1.0 - xi_3)),
            ],
            [
                0.125
                    * (-(1.0 - xi_2) * (1.0 + xi_3) * (-xi_1 - xi_2 + xi_3 - 2.0)
                        - (1.0 - xi_1) * (1.0 - xi_2) * (1.0 + xi_3)),
                0.125
                    * (-(1.0 - xi_1) * (1.0 + xi_3) * (-xi_1 - xi_2 + xi_3 - 2.0)
                        - (1.0 - xi_1) * (1.0 - xi_2) * (1.0 + xi_3)),
                0.125
                    * ((1.0 - xi_1) * (1.0 - xi_2) * (-xi_1 - xi_2 + xi_3 - 2.0)
                        + (1.0 - xi_1) * (1.0 - xi_2) * (1.0 + xi_3)),
            ],
            [
                0.125
                    * ((1.0 - xi_2) * (1.0 + xi_3) * (xi_1 - xi_2 + xi_3 - 2.0)
                        + (1.0 + xi_1) * (1.0 - xi_2) * (1.0 + xi_3)),
                0.125
                    * (-(1.0 + xi_1) * (1.0 + xi_3) * (xi_1 - xi_2 + xi_3 - 2.0)
                        - (1.0 + xi_1) * (1.0 - xi_2) * (1.0 + xi_3)),
                0.125
                    * ((1.0 + xi_1) * (1.0 - xi_2) * (xi_1 - xi_2 + xi_3 - 2.0)
                        + (1.0 + xi_1) * (1.0 - xi_2) * (1.0 + xi_3)),
            ],
            [
                0.125
                    * ((1.0 + xi_2) * (1.0 + xi_3) * (xi_1 + xi_2 + xi_3 - 2.0)
                        + (1.0 + xi_1) * (1.0 + xi_2) * (1.0 + xi_3)),
                0.125
                    * ((1.0 + xi_1) * (1.0 + xi_3) * (xi_1 + xi_2 + xi_3 - 2.0)
                        + (1.0 + xi_1) * (1.0 + xi_2) * (1.0 + xi_3)),
                0.125
                    * ((1.0 + xi_1) * (1.0 + xi_2) * (xi_1 + xi_2 + xi_3 - 2.0)
                        + (1.0 + xi_1) * (1.0 + xi_2) * (1.0 + xi_3)),
            ],
            [
                0.125
                    * (-(1.0 + xi_2) * (1.0 + xi_3) * (-xi_1 + xi_2 + xi_3 - 2.0)
                        - (1.0 - xi_1) * (1.0 + xi_2) * (1.0 + xi_3)),
                0.125
                    * ((1.0 - xi_1) * (1.0 + xi_3) * (-xi_1 + xi_2 + xi_3 - 2.0)
                        + (1.0 - xi_1) * (1.0 + xi_2) * (1.0 + xi_3)),
                0.125
                    * ((1.0 - xi_1) * (1.0 + xi_2) * (-xi_1 + xi_2 + xi_3 - 2.0)
                        + (1.0 - xi_1) * (1.0 + xi_2) * (1.0 + xi_3)),
            ],
            [
                -0.5 * xi_1 * (1.0 - xi_2) * (1.0 - xi_3),
                -0.25 * (1.0 - xi_1 * xi_1) * (1.0 - xi_3),
                -0.25 * (1.0 - xi_1 * xi_1) * (1.0 - xi_2),
            ],
            [
                0.25 * (1.0 - xi_2 * xi_2) * (1.0 - xi_3),
                -0.5 * xi_2 * (1.0 + xi_1) * (1.0 - xi_3),
                -0.25 * (1.0 + xi_1) * (1.0 - xi_2 * xi_2),
            ],
            [
                -0.5 * xi_1 * (1.0 + xi_2) * (1.0 - xi_3),
                0.25 * (1.0 - xi_1 * xi_1) * (1.0 - xi_3),
                -0.25 * (1.0 - xi_1 * xi_1) * (1.0 + xi_2),
            ],
            [
                -0.25 * (1.0 - xi_2 * xi_2) * (1.0 - xi_3),
                -0.5 * xi_2 * (1.0 - xi_1) * (1.0 - xi_3),
                -0.25 * (1.0 - xi_1) * (1.0 - xi_2 * xi_2),
            ],
            [
                -0.25 * (1.0 - xi_2) * (1.0 - xi_3 * xi_3),
                -0.25 * (1.0 - xi_1) * (1.0 - xi_3 * xi_3),
                -0.5 * xi_3 * (1.0 - xi_1) * (1.0 - xi_2),
            ],
            [
                0.25 * (1.0 - xi_2) * (1.0 - xi_3 * xi_3),
                -0.25 * (1.0 + xi_1) * (1.0 - xi_3 * xi_3),
                -0.5 * xi_3 * (1.0 + xi_1) * (1.0 - xi_2),
            ],
            [
                0.25 * (1.0 + xi_2) * (1.0 - xi_3 * xi_3),
                0.25 * (1.0 + xi_1) * (1.0 - xi_3 * xi_3),
                -0.5 * xi_3 * (1.0 + xi_1) * (1.0 + xi_2),
            ],
            [
                -0.25 * (1.0 + xi_2) * (1.0 - xi_3 * xi_3),
                0.25 * (1.0 - xi_1) * (1.0 - xi_3 * xi_3),
                -0.5 * xi_3 * (1.0 - xi_1) * (1.0 + xi_2),
            ],
            [
                -0.5 * xi_1 * (1.0 - xi_2) * (1.0 + xi_3),
                -0.25 * (1.0 - xi_1 * xi_1) * (1.0 + xi_3),
                0.25 * (1.0 - xi_1 * xi_1) * (1.0 - xi_2),
            ],
            [
                0.25 * (1.0 - xi_2 * xi_2) * (1.0 + xi_3),
                -0.5 * xi_2 * (1.0 + xi_1) * (1.0 + xi_3),
                0.25 * (1.0 + xi_1) * (1.0 - xi_2 * xi_2),
            ],
            [
                -0.5 * xi_1 * (1.0 + xi_2) * (1.0 + xi_3),
                0.25 * (1.0 - xi_1 * xi_1) * (1.0 + xi_3),
                0.25 * (1.0 - xi_1 * xi_1) * (1.0 + xi_2),
            ],
            [
                -0.25 * (1.0 - xi_2 * xi_2) * (1.0 + xi_3),
                -0.5 * xi_2 * (1.0 - xi_1) * (1.0 + xi_3),
                0.25 * (1.0 - xi_1) * (1.0 - xi_2 * xi_2),
            ],
        ]
        .into()
    }
}

impl QuadraticFiniteElement<G, N> for Hexahedron {}
