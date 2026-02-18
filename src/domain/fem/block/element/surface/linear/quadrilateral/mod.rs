#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        ElementNodalEitherCoordinates, FRAC_1_SQRT_3, FiniteElement, ParametricCoordinate,
        ParametricCoordinates, ParametricReference, ShapeFunctions, ShapeFunctionsGradients,
        surface::{M, linear::LinearSurfaceElement},
    },
    math::{ScalarList, Tensor, TensorArray},
    mechanics::Coordinate,
};

const G: usize = 4;
const N: usize = 4;
const P: usize = N;

const CORNERS: [[usize; 2]; N] = [[1, 3], [2, 0], [3, 1], [0, 2]];

pub type Quadrilateral = LinearSurfaceElement<G, N>;

impl FiniteElement<G, M, N, P> for Quadrilateral {
    fn integration_points() -> ParametricCoordinates<G, M> {
        [
            [-FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
            [FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
            [FRAC_1_SQRT_3, FRAC_1_SQRT_3],
            [-FRAC_1_SQRT_3, FRAC_1_SQRT_3],
        ]
        .into()
    }
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]].into()
    }
    fn parametric_weights() -> ScalarList<G> {
        [1.0; G].into()
    }
    fn scaled_jacobians<const I: usize>(
        nodal_coordinates: ElementNodalEitherCoordinates<I, N>,
    ) -> ScalarList<P> {
        let mut u = Coordinate::zero();
        let mut v = Coordinate::zero();
        let mut n = Coordinate::zero();
        let x = (&nodal_coordinates[1] - &nodal_coordinates[0])
            + (&nodal_coordinates[2] - &nodal_coordinates[3]);
        let y = (&nodal_coordinates[2] - &nodal_coordinates[1])
            + (&nodal_coordinates[3] - &nodal_coordinates[0]);
        let nc = x.cross(&y).normalized();
        CORNERS
            .into_iter()
            .enumerate()
            .map(|(node, [node_a, node_b])| {
                u = &nodal_coordinates[node_a] - &nodal_coordinates[node];
                v = &nodal_coordinates[node_b] - &nodal_coordinates[node];
                n = u.cross(&v);
                (&n * &nc) / u.norm() / v.norm()
            })
            .collect()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2] = parametric_coordinate.into();
        [
            (1.0 - xi_1) * (1.0 - xi_2) / 4.0,
            (1.0 + xi_1) * (1.0 - xi_2) / 4.0,
            (1.0 + xi_1) * (1.0 + xi_2) / 4.0,
            (1.0 - xi_1) * (1.0 + xi_2) / 4.0,
        ]
        .into()
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        let [xi_1, xi_2] = parametric_coordinate.into();
        [
            [-(1.0 - xi_2) / 4.0, -(1.0 - xi_1) / 4.0],
            [(1.0 - xi_2) / 4.0, -(1.0 + xi_1) / 4.0],
            [(1.0 + xi_2) / 4.0, (1.0 + xi_1) / 4.0],
            [-(1.0 + xi_2) / 4.0, (1.0 - xi_1) / 4.0],
        ]
        .into()
    }
}
