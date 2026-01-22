#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        ElementNodalCoordinates, FRAC_SQRT_3_5, FiniteElement, ParametricCoordinate,
        ParametricCoordinates, ParametricReference, ShapeFunctions, ShapeFunctionsGradients,
        quadratic::{M, QuadraticElement, QuadraticFiniteElement},
    },
    math::{Scalar, ScalarList, TensorRank1},
};

const G: usize = 27;
const N: usize = 27;
const P: usize = N;

pub type Hexahedron = QuadraticElement<G, N>;

impl FiniteElement<G, M, N, P> for Hexahedron {
    fn integration_points() -> ParametricCoordinates<G, M> {
        const POINTS: [Scalar; 3] = [-FRAC_SQRT_3_5, 0.0, FRAC_SQRT_3_5];
        POINTS
            .into_iter()
            .flat_map(|z| {
                POINTS.into_iter().flat_map(move |y| {
                    POINTS
                        .into_iter()
                        .map(move |x| TensorRank1::from([x, y, z]))
                })
            })
            .collect()
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
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        .into()
    }
    fn parametric_weights() -> ScalarList<G> {
        const WEIGHTS: [Scalar; 3] = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0];
        WEIGHTS
            .into_iter()
            .flat_map(|wz| {
                WEIGHTS
                    .into_iter()
                    .flat_map(move |wy| WEIGHTS.into_iter().map(move |wx| wx * wy * wz))
            })
            .collect()
    }
    fn scaled_jacobians(_nodal_coordinates: &ElementNodalCoordinates<N>) -> ScalarList<P> {
        todo!()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        [
            -0.125 * xi_1 * xi_2 * xi_3 * (1.0 - xi_1) * (1.0 - xi_2) * (1.0 - xi_3),
            0.125 * xi_1 * xi_2 * xi_3 * (1.0 + xi_1) * (1.0 - xi_2) * (1.0 - xi_3),
            -0.125 * xi_1 * xi_2 * xi_3 * (1.0 + xi_1) * (1.0 + xi_2) * (1.0 - xi_3),
            0.125 * xi_1 * xi_2 * xi_3 * (1.0 - xi_1) * (1.0 + xi_2) * (1.0 - xi_3),
            0.125 * xi_1 * xi_2 * xi_3 * (1.0 - xi_1) * (1.0 - xi_2) * (1.0 + xi_3),
            -0.125 * xi_1 * xi_2 * xi_3 * (1.0 + xi_1) * (1.0 - xi_2) * (1.0 + xi_3),
            0.125 * xi_1 * xi_2 * xi_3 * (1.0 + xi_1) * (1.0 + xi_2) * (1.0 + xi_3),
            -0.125 * xi_1 * xi_2 * xi_3 * (1.0 - xi_1) * (1.0 + xi_2) * (1.0 + xi_3),
            0.25 * xi_2 * xi_3 * (1.0 - xi_1 * xi_1) * (1.0 - xi_2) * (1.0 - xi_3),
            -0.25 * xi_1 * xi_3 * (1.0 + xi_1) * (1.0 - xi_2 * xi_2) * (1.0 - xi_3),
            -0.25 * xi_2 * xi_3 * (1.0 - xi_1 * xi_1) * (1.0 + xi_2) * (1.0 - xi_3),
            0.25 * xi_1 * xi_3 * (1.0 - xi_1) * (1.0 - xi_2 * xi_2) * (1.0 - xi_3),
            0.25 * xi_1 * xi_2 * (1.0 - xi_1) * (1.0 - xi_2) * (1.0 - xi_3 * xi_3),
            -0.25 * xi_1 * xi_2 * (1.0 + xi_1) * (1.0 - xi_2) * (1.0 - xi_3 * xi_3),
            0.25 * xi_1 * xi_2 * (1.0 + xi_1) * (1.0 + xi_2) * (1.0 - xi_3 * xi_3),
            -0.25 * xi_1 * xi_2 * (1.0 - xi_1) * (1.0 + xi_2) * (1.0 - xi_3 * xi_3),
            -0.25 * xi_2 * xi_3 * (1.0 - xi_1 * xi_1) * (1.0 - xi_2) * (1.0 + xi_3),
            0.25 * xi_1 * xi_3 * (1.0 + xi_1) * (1.0 - xi_2 * xi_2) * (1.0 + xi_3),
            0.25 * xi_2 * xi_3 * (1.0 - xi_1 * xi_1) * (1.0 + xi_2) * (1.0 + xi_3),
            -0.25 * xi_1 * xi_3 * (1.0 - xi_1) * (1.0 - xi_2 * xi_2) * (1.0 + xi_3),
            (1.0 - xi_1 * xi_1) * (1.0 - xi_2 * xi_2) * (1.0 - xi_3 * xi_3),
            -0.5 * xi_3 * (1.0 - xi_1 * xi_1) * (1.0 - xi_2 * xi_2) * (1.0 - xi_3),
            0.5 * xi_3 * (1.0 - xi_1 * xi_1) * (1.0 - xi_2 * xi_2) * (1.0 + xi_3),
            -0.5 * xi_1 * (1.0 - xi_1) * (1.0 - xi_2 * xi_2) * (1.0 - xi_3 * xi_3),
            0.5 * xi_1 * (1.0 + xi_1) * (1.0 - xi_2 * xi_2) * (1.0 - xi_3 * xi_3),
            -0.5 * xi_2 * (1.0 - xi_1 * xi_1) * (1.0 - xi_2) * (1.0 - xi_3 * xi_3),
            0.5 * xi_2 * (1.0 - xi_1 * xi_1) * (1.0 + xi_2) * (1.0 - xi_3 * xi_3),
        ]
        .into()
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        [
            [
                -0.125 * xi_2 * xi_3 * (1.0 - 2.0 * xi_1) * (1.0 - xi_2) * (1.0 - xi_3),
                -0.125 * xi_1 * xi_3 * (1.0 - xi_1) * (1.0 - 2.0 * xi_2) * (1.0 - xi_3),
                -0.125 * xi_1 * xi_2 * (1.0 - xi_1) * (1.0 - xi_2) * (1.0 - 2.0 * xi_3),
            ],
            [
                0.125 * xi_2 * xi_3 * (1.0 + 2.0 * xi_1) * (1.0 - xi_2) * (1.0 - xi_3),
                0.125 * xi_1 * xi_3 * (1.0 + xi_1) * (1.0 - 2.0 * xi_2) * (1.0 - xi_3),
                0.125 * xi_1 * xi_2 * (1.0 + xi_1) * (1.0 - xi_2) * (1.0 - 2.0 * xi_3),
            ],
            [
                -0.125 * xi_2 * xi_3 * (1.0 + 2.0 * xi_1) * (1.0 + xi_2) * (1.0 - xi_3),
                -0.125 * xi_1 * xi_3 * (1.0 + xi_1) * (1.0 + 2.0 * xi_2) * (1.0 - xi_3),
                -0.125 * xi_1 * xi_2 * (1.0 + xi_1) * (1.0 + xi_2) * (1.0 - 2.0 * xi_3),
            ],
            [
                0.125 * xi_2 * xi_3 * (1.0 - 2.0 * xi_1) * (1.0 + xi_2) * (1.0 - xi_3),
                0.125 * xi_1 * xi_3 * (1.0 - xi_1) * (1.0 + 2.0 * xi_2) * (1.0 - xi_3),
                0.125 * xi_1 * xi_2 * (1.0 - xi_1) * (1.0 + xi_2) * (1.0 - 2.0 * xi_3),
            ],
            [
                0.125 * xi_2 * xi_3 * (1.0 - 2.0 * xi_1) * (1.0 - xi_2) * (1.0 + xi_3),
                0.125 * xi_1 * xi_3 * (1.0 - xi_1) * (1.0 - 2.0 * xi_2) * (1.0 + xi_3),
                0.125 * xi_1 * xi_2 * (1.0 - xi_1) * (1.0 - xi_2) * (1.0 + 2.0 * xi_3),
            ],
            [
                -0.125 * xi_2 * xi_3 * (1.0 + 2.0 * xi_1) * (1.0 - xi_2) * (1.0 + xi_3),
                -0.125 * xi_1 * xi_3 * (1.0 + xi_1) * (1.0 - 2.0 * xi_2) * (1.0 + xi_3),
                -0.125 * xi_1 * xi_2 * (1.0 + xi_1) * (1.0 - xi_2) * (1.0 + 2.0 * xi_3),
            ],
            [
                0.125 * xi_2 * xi_3 * (1.0 + 2.0 * xi_1) * (1.0 + xi_2) * (1.0 + xi_3),
                0.125 * xi_1 * xi_3 * (1.0 + xi_1) * (1.0 + 2.0 * xi_2) * (1.0 + xi_3),
                0.125 * xi_1 * xi_2 * (1.0 + xi_1) * (1.0 + xi_2) * (1.0 + 2.0 * xi_3),
            ],
            [
                -0.125 * xi_2 * xi_3 * (1.0 - 2.0 * xi_1) * (1.0 + xi_2) * (1.0 + xi_3),
                -0.125 * xi_1 * xi_3 * (1.0 - xi_1) * (1.0 + 2.0 * xi_2) * (1.0 + xi_3),
                -0.125 * xi_1 * xi_2 * (1.0 - xi_1) * (1.0 + xi_2) * (1.0 + 2.0 * xi_3),
            ],
            [
                -0.5 * xi_1 * xi_2 * xi_3 * (1.0 - xi_2) * (1.0 - xi_3),
                0.25 * xi_3 * (1.0 - xi_1 * xi_1) * (1.0 - 2.0 * xi_2) * (1.0 - xi_3),
                0.25 * xi_2 * (1.0 - xi_1 * xi_1) * (1.0 - xi_2) * (1.0 - 2.0 * xi_3),
            ],
            [
                -0.25 * xi_3 * (1.0 + 2.0 * xi_1) * (1.0 - xi_2 * xi_2) * (1.0 - xi_3),
                0.5 * xi_1 * xi_2 * xi_3 * (1.0 + xi_1) * (1.0 - xi_3),
                -0.25 * xi_1 * (1.0 + xi_1) * (1.0 - xi_2 * xi_2) * (1.0 - 2.0 * xi_3),
            ],
            [
                0.5 * xi_1 * xi_2 * xi_3 * (1.0 + xi_2) * (1.0 - xi_3),
                -0.25 * xi_3 * (1.0 - xi_1 * xi_1) * (1.0 + 2.0 * xi_2) * (1.0 - xi_3),
                -0.25 * xi_2 * (1.0 - xi_1 * xi_1) * (1.0 + xi_2) * (1.0 - 2.0 * xi_3),
            ],
            [
                0.25 * xi_3 * (1.0 - 2.0 * xi_1) * (1.0 - xi_2 * xi_2) * (1.0 - xi_3),
                -0.5 * xi_1 * xi_2 * xi_3 * (1.0 - xi_1) * (1.0 - xi_3),
                0.25 * xi_1 * (1.0 - xi_1) * (1.0 - xi_2 * xi_2) * (1.0 - 2.0 * xi_3),
            ],
            [
                0.25 * xi_2 * (1.0 - 2.0 * xi_1) * (1.0 - xi_2) * (1.0 - xi_3 * xi_3),
                0.25 * xi_1 * (1.0 - xi_1) * (1.0 - 2.0 * xi_2) * (1.0 - xi_3 * xi_3),
                -0.5 * xi_1 * xi_2 * xi_3 * (1.0 - xi_1) * (1.0 - xi_2),
            ],
            [
                -0.25 * xi_2 * (1.0 + 2.0 * xi_1) * (1.0 - xi_2) * (1.0 - xi_3 * xi_3),
                -0.25 * xi_1 * (1.0 + xi_1) * (1.0 - 2.0 * xi_2) * (1.0 - xi_3 * xi_3),
                0.5 * xi_1 * xi_2 * xi_3 * (1.0 + xi_1) * (1.0 - xi_2),
            ],
            [
                0.25 * xi_2 * (1.0 + 2.0 * xi_1) * (1.0 + xi_2) * (1.0 - xi_3 * xi_3),
                0.25 * xi_1 * (1.0 + xi_1) * (1.0 + 2.0 * xi_2) * (1.0 - xi_3 * xi_3),
                -0.5 * xi_1 * xi_2 * xi_3 * (1.0 + xi_1) * (1.0 + xi_2),
            ],
            [
                -0.25 * xi_2 * (1.0 - 2.0 * xi_1) * (1.0 + xi_2) * (1.0 - xi_3 * xi_3),
                -0.25 * xi_1 * (1.0 - xi_1) * (1.0 + 2.0 * xi_2) * (1.0 - xi_3 * xi_3),
                0.5 * xi_1 * xi_2 * xi_3 * (1.0 - xi_1) * (1.0 + xi_2),
            ],
            [
                0.5 * xi_1 * xi_2 * xi_3 * (1.0 - xi_2) * (1.0 + xi_3),
                -0.25 * xi_3 * (1.0 - xi_1 * xi_1) * (1.0 - 2.0 * xi_2) * (1.0 + xi_3),
                -0.25 * xi_2 * (1.0 - xi_1 * xi_1) * (1.0 - xi_2) * (1.0 + 2.0 * xi_3),
            ],
            [
                0.25 * xi_3 * (1.0 + 2.0 * xi_1) * (1.0 - xi_2 * xi_2) * (1.0 + xi_3),
                -0.5 * xi_1 * xi_2 * xi_3 * (1.0 + xi_1) * (1.0 + xi_3),
                0.25 * xi_1 * (1.0 + xi_1) * (1.0 - xi_2 * xi_2) * (1.0 + 2.0 * xi_3),
            ],
            [
                -0.5 * xi_1 * xi_2 * xi_3 * (1.0 + xi_2) * (1.0 + xi_3),
                0.25 * xi_3 * (1.0 - xi_1 * xi_1) * (1.0 + 2.0 * xi_2) * (1.0 + xi_3),
                0.25 * xi_2 * (1.0 - xi_1 * xi_1) * (1.0 + xi_2) * (1.0 + 2.0 * xi_3),
            ],
            [
                -0.25 * xi_3 * (1.0 - 2.0 * xi_1) * (1.0 - xi_2 * xi_2) * (1.0 + xi_3),
                0.5 * xi_1 * xi_2 * xi_3 * (1.0 - xi_1) * (1.0 + xi_3),
                -0.25 * xi_1 * (1.0 - xi_1) * (1.0 - xi_2 * xi_2) * (1.0 + 2.0 * xi_3),
            ],
            [
                -2.0 * xi_1 * (1.0 - xi_2 * xi_2) * (1.0 - xi_3 * xi_3),
                -2.0 * xi_2 * (1.0 - xi_1 * xi_1) * (1.0 - xi_3 * xi_3),
                -2.0 * xi_3 * (1.0 - xi_1 * xi_1) * (1.0 - xi_2 * xi_2),
            ],
            [
                xi_1 * xi_3 * (1.0 - xi_2 * xi_2) * (1.0 - xi_3),
                xi_2 * xi_3 * (1.0 - xi_1 * xi_1) * (1.0 - xi_3),
                -0.5 * (1.0 - xi_1 * xi_1) * (1.0 - xi_2 * xi_2) * (1.0 - 2.0 * xi_3),
            ],
            [
                -xi_1 * xi_3 * (1.0 - xi_2 * xi_2) * (1.0 + xi_3),
                -xi_2 * xi_3 * (1.0 - xi_1 * xi_1) * (1.0 + xi_3),
                0.5 * (1.0 - xi_1 * xi_1) * (1.0 - xi_2 * xi_2) * (1.0 + 2.0 * xi_3),
            ],
            [
                -0.5 * (1.0 - 2.0 * xi_1) * (1.0 - xi_2 * xi_2) * (1.0 - xi_3 * xi_3),
                xi_1 * xi_2 * (1.0 - xi_1) * (1.0 - xi_3 * xi_3),
                xi_1 * xi_3 * (1.0 - xi_1) * (1.0 - xi_2 * xi_2),
            ],
            [
                0.5 * (1.0 + 2.0 * xi_1) * (1.0 - xi_2 * xi_2) * (1.0 - xi_3 * xi_3),
                -xi_1 * xi_2 * (1.0 + xi_1) * (1.0 - xi_3 * xi_3),
                -xi_1 * xi_3 * (1.0 + xi_1) * (1.0 - xi_2 * xi_2),
            ],
            [
                xi_1 * xi_2 * (1.0 - xi_2) * (1.0 - xi_3 * xi_3),
                -0.5 * (1.0 - xi_1 * xi_1) * (1.0 - 2.0 * xi_2) * (1.0 - xi_3 * xi_3),
                xi_2 * xi_3 * (1.0 - xi_1 * xi_1) * (1.0 - xi_2),
            ],
            [
                -xi_1 * xi_2 * (1.0 + xi_2) * (1.0 - xi_3 * xi_3),
                0.5 * (1.0 - xi_1 * xi_1) * (1.0 + 2.0 * xi_2) * (1.0 - xi_3 * xi_3),
                -xi_2 * xi_3 * (1.0 - xi_1 * xi_1) * (1.0 + xi_2),
            ],
        ]
        .into()
    }
}

impl QuadraticFiniteElement<G, N> for Hexahedron {}
