#[cfg(test)]
pub mod test;

use crate::{
    fem::block::element::{
        FiniteElement, ParametricCoordinate, ParametricCoordinates, ParametricReference,
        ShapeFunctions, ShapeFunctionsGradients,
        quadratic::{M, QuadraticElement, QuadraticFiniteElement},
    },
    math::ScalarList,
};

const G: usize = 4;
const N: usize = 10;

pub type Tetrahedron = QuadraticElement<G, N>;

impl FiniteElement<G, M, N> for Tetrahedron {
    fn integration_points() -> ParametricCoordinates<G, M> {
        let alpha = (1.0 + 3.0 / 5.0_f64.sqrt()) / 4.0;
        let beta = (1.0 - 1.0 / 5.0_f64.sqrt()) / 4.0;
        [
            [beta, beta, beta],
            [alpha, beta, beta],
            [beta, alpha, beta],
            [beta, beta, alpha],
        ]
        .into()
    }
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
        .into()
    }
    fn parametric_weights() -> ScalarList<G> {
        [1.0 / 24.0; G].into()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        let xi_0 = 1.0 - xi_1 - xi_2 - xi_3;
        [
            xi_0 * (2.0 * xi_0 - 1.0),
            xi_1 * (2.0 * xi_1 - 1.0),
            xi_2 * (2.0 * xi_2 - 1.0),
            xi_3 * (2.0 * xi_3 - 1.0),
            4.0 * xi_0 * xi_1,
            4.0 * xi_1 * xi_2,
            4.0 * xi_0 * xi_2,
            4.0 * xi_0 * xi_3,
            4.0 * xi_1 * xi_3,
            4.0 * xi_2 * xi_3,
        ]
        .into()
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        let xi_0 = 1.0 - xi_1 - xi_2 - xi_3;
        [
            [-4.0 * xi_0 + 1.0, -4.0 * xi_0 + 1.0, -4.0 * xi_0 + 1.0],
            [4.0 * xi_1 - 1.0, 0.0, 0.0],
            [0.0, 4.0 * xi_2 - 1.0, 0.0],
            [0.0, 0.0, 4.0 * xi_3 - 1.0],
            [4.0 * (xi_0 - xi_1), -4.0 * xi_1, -4.0 * xi_1],
            [4.0 * xi_2, 4.0 * xi_1, 0.0],
            [-4.0 * xi_2, 4.0 * (xi_0 - xi_2), -4.0 * xi_2],
            [-4.0 * xi_3, -4.0 * xi_3, 4.0 * (xi_0 - xi_3)],
            [4.0 * xi_3, 0.0, 4.0 * xi_1],
            [0.0, 4.0 * xi_3, 4.0 * xi_2],
        ]
        .into()
    }
}

impl QuadraticFiniteElement<G, N> for Tetrahedron {}
