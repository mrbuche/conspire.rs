#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        ElementNodalEitherCoordinates, FRAC_1_SQRT_3, FiniteElement, ParametricCoordinate,
        ParametricCoordinates, ParametricReference, ShapeFunctions, ShapeFunctionsGradients,
        linear::{LinearElement, LinearFiniteElement, M},
    },
    math::{ScalarList, Tensor, TensorArray},
    mechanics::Coordinate,
};

const G: usize = 8;
const N: usize = 8;
const P: usize = N;

const CORNERS: [[usize; 3]; N] = [
    [1, 3, 4],
    [2, 0, 5],
    [3, 1, 6],
    [0, 2, 7],
    [7, 5, 0],
    [4, 6, 1],
    [5, 7, 2],
    [6, 4, 3],
];

pub type Hexahedron = LinearElement<G, N>;

impl FiniteElement<G, M, N, P> for Hexahedron {
    fn integration_points() -> ParametricCoordinates<G, M> {
        [
            [-FRAC_1_SQRT_3, -FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
            [-FRAC_1_SQRT_3, -FRAC_1_SQRT_3, FRAC_1_SQRT_3],
            [-FRAC_1_SQRT_3, FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
            [-FRAC_1_SQRT_3, FRAC_1_SQRT_3, FRAC_1_SQRT_3],
            [FRAC_1_SQRT_3, -FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
            [FRAC_1_SQRT_3, -FRAC_1_SQRT_3, FRAC_1_SQRT_3],
            [FRAC_1_SQRT_3, FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
            [FRAC_1_SQRT_3, FRAC_1_SQRT_3, FRAC_1_SQRT_3],
        ]
        .into()
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
        ]
        .into()
    }
    fn parametric_weights() -> ScalarList<G> {
        [1.0; G].into()
    }
    fn scaled_jacobians<const I: usize>(
        nodal_coordinates: ElementNodalEitherCoordinates<I, N>,
    ) -> ScalarList<P> {
        let mut u = Coordinate::zero();
        let mut v = Coordinate::zero();
        let mut w = Coordinate::zero();
        let mut n = Coordinate::zero();
        CORNERS
            .into_iter()
            .enumerate()
            .map(|(node, [node_a, node_b, node_c])| {
                u = &nodal_coordinates[node_a] - &nodal_coordinates[node];
                v = &nodal_coordinates[node_b] - &nodal_coordinates[node];
                w = &nodal_coordinates[node_c] - &nodal_coordinates[node];
                n = u.cross(&v);
                (&n * &w) / u.norm() / v.norm() / w.norm()
            })
            .collect()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
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
        .into()
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
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
        .into()
    }
}

impl LinearFiniteElement<G, N> for Hexahedron {}
