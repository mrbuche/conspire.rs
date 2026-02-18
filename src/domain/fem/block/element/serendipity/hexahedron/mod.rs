#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        ElementNodalEitherCoordinates, FiniteElement, ParametricCoordinate, ParametricCoordinates,
        ParametricReference, ShapeFunctions, ShapeFunctionsGradients,
        quadratic::{Hexahedron as QuadraticHexahedron, QuadraticElement, QuadraticFiniteElement},
        serendipity::M,
    },
    math::ScalarList,
};

const G: usize = 27;
const N: usize = 20;
const P: usize = N;

pub type Hexahedron = QuadraticElement<G, N>;

impl FiniteElement<G, M, N, P> for Hexahedron {
    fn integration_points() -> ParametricCoordinates<G, M> {
        QuadraticHexahedron::integration_points()
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
        QuadraticHexahedron::parametric_weights()
    }
    fn scaled_jacobians<const I: usize>(
        _nodal_coordinates: ElementNodalEitherCoordinates<I, N>,
    ) -> ScalarList<P> {
        todo!()
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
