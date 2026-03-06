#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        FRAC_1_SQRT_3, FiniteElement, ParametricCoordinate, ParametricCoordinates,
        ParametricReference, ShapeFunctions, ShapeFunctionsGradients,
        linear::{LinearElement, LinearFiniteElement, M},
    },
    math::ScalarList,
};

const G: usize = 6;
const N: usize = 6;
const P: usize = N;

pub type Wedge = LinearElement<G, N>;

impl FiniteElement<G, M, N, P> for Wedge {
    fn integration_points() -> ParametricCoordinates<G, M> {
        [
            [2.0 / 3.0, 1.0 / 6.0, -FRAC_1_SQRT_3],
            [1.0 / 6.0, 2.0 / 3.0, -FRAC_1_SQRT_3],
            [1.0 / 6.0, 1.0 / 6.0, -FRAC_1_SQRT_3],
            [2.0 / 3.0, 1.0 / 6.0, FRAC_1_SQRT_3],
            [1.0 / 6.0, 2.0 / 3.0, FRAC_1_SQRT_3],
            [1.0 / 6.0, 1.0 / 6.0, FRAC_1_SQRT_3],
        ]
        .into()
    }
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
        .into()
    }
    fn parametric_weights() -> ScalarList<G> {
        [1.0 / 6.0; G].into()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        let xi_0 = 1.0 - xi_1 - xi_2;
        [
            0.5 * xi_0 * (1.0 - xi_3),
            0.5 * xi_1 * (1.0 - xi_3),
            0.5 * xi_2 * (1.0 - xi_3),
            0.5 * xi_0 * (1.0 + xi_3),
            0.5 * xi_1 * (1.0 + xi_3),
            0.5 * xi_2 * (1.0 + xi_3),
        ]
        .into()
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        let xi_0 = 1.0 - xi_1 - xi_2;
        [
            [-0.5 * (1.0 - xi_3), -0.5 * (1.0 - xi_3), -0.5 * xi_0],
            [0.5 * (1.0 - xi_3), 0.0, -0.5 * xi_1],
            [0.0, 0.5 * (1.0 - xi_3), -0.5 * xi_2],
            [-0.5 * (1.0 + xi_3), -0.5 * (1.0 + xi_3), 0.5 * xi_0],
            [0.5 * (1.0 + xi_3), 0.0, 0.5 * xi_1],
            [0.0, 0.5 * (1.0 + xi_3), 0.5 * xi_2],
        ]
        .into()
    }
}

impl LinearFiniteElement<G, N> for Wedge {}
